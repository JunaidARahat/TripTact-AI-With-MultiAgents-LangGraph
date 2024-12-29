from flask import Flask, render_template, request, jsonify
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Initialize Flask app
app = Flask(__name__)

# Define the PlannerState class
class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    city: str
    interests: List[str]
    recommendations: List[str]
    itinerary: str

# Define the LLM for agents
city_agent = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile"
)

interest_agent = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile"
)

itinerary_agent = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile"
)

# Define prompts for each agent
city_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a city expert. Recommend activities and locations in {city} for a day trip."),
    ("human", "What are the best places to visit in {city}?")
])

interest_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an interest parser. Categorize the user's interests: {interests} into activities like museums, parks, or cafes."),
    ("human", "Categorize my interests.")
])

itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} using these recommendations: {recommendations} and these interests: {interests}. Provide a brief, bulleted itinerary."),
    ("human", "Create a complete itinerary for my trip.")
])

# Input processing functions
def get_city_recommendations(city: str) -> List[str]:
    response = city_agent.invoke(city_prompt.format_messages(city=city))
    return response.content.split("\n")  # Split response into list of recommendations

def parse_interests(interests: str) -> List[str]:
    response = interest_agent.invoke(interest_prompt.format_messages(interests=interests))
    return [interest.strip() for interest in response.content.split(",")]

def create_itinerary(city: str, recommendations: List[str], interests: List[str]) -> str:
    response = itinerary_agent.invoke(itinerary_prompt.format_messages(
        city=city, recommendations="; ".join(recommendations), interests=", ".join(interests)))
    return response.content

# Flask routes
@app.route("/", methods=["GET", "POST"])
def travel_planner():
    if request.method == "POST":
        # Get user input from form
        city = request.form.get("city")
        interests = request.form.get("interests")

        if not city or not interests:
            return render_template("index.html", error="Please fill in both fields.")

        # Process input through agents
        recommendations = get_city_recommendations(city)
        parsed_interests = parse_interests(interests)
        itinerary = create_itinerary(city, recommendations, parsed_interests)

        # Render the result
        return render_template("index.html", city=city, interests=interests, itinerary=itinerary)

    return render_template("index.html")

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
 