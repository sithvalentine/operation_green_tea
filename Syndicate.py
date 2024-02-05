import os

from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import ollama


# To Load Local models through Ollama
wizard_vicuna_uncensored = ollama(model="wizard-vicuna-uncensored")

# To Load GPT-4
api_openai = os.environ.get("OPENAI_API_KEY")

#To Load gemini
api_gemini = os.environ.get("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.1, google_api_key=api_gemini
)

researcher = Agent(
   role="Research Analyst",
   goal="Research new ecotourism insights for a blog so two articles per week can be written. Share what you find with the writer so the blogs can be written. Also give your personal input on the information so the writer can have enough information to write on.",
   backstory="You have a bachelor's of science degree from Tulane Univeristy in Linguistics. You are travel researcher that specializes in ecotourism and you have experience traveling to all 195 countries",
   verbose=True, # enable more detailed pr extensive output
   allow_delegation=True, # enable collaboration between agents
   #llm=llm  to load gemini

)

travel_writer = Agent(
    role="Travel writer",
    goal="Write two intriguing entries in a travel blog and a newsletter that focuses on ecotourism using your experience. For each blog, use proper Title, Headings, internal links, and backlinks with the help from the SEO specialist.Each blog should be at least 1000 words.",
    backstory="""Your education is is a bachelor's of science degree from Tulane University in economics and a minor in Environmental Studies. You have experience traveling throughout the United States including many countries in Europe, Africa, and Asia. 
                You have traveled with the researcher for more than ten years and now you two work together to produce content for the blog jamesburnsmedia.com. Also, write two LinkedIN newsletters per week with 200 to 500 words.""",
    verbose=False, # enable more detailed or extensive output
    allow_delegation=True, # enable collaboration between agents
# llm=llm  To load gemini
)

seo_specialist = Agent(
    role="SEO Expert",
    goal="Test, analyze, and modify jamesburnsmedia.com so it is optimized for search engines, and the website subsequently ranks higher in the search results on major search engines such as Google and Bing in the ecotourism niche. Be sure to use proper Titles, headings, internal links, and backlinks for best optimization.",
    backstory="You have a bachelor's degree in marketing from UC Davis, a Google Fundamentals of Digital marketing Certification, and an ahrefs certification. You have been employed by major corporations such as Netflix and Warner Bros. Discovery as well as a few local companies.",
    verbose=True

)
task1 = Task(
    description="""Analyze the travel market for ecotourism. Find out all aspects of travel in the ecotourism niche by finding out good places to travel, bad places to travel. Find the best countries for ecotourism and the worst places for ecotourism. 
                Find out itineraries for all countries where ecotourism can be done. Find the best places to see wildlife. Find out about places travelers can volunteer while on a tour. Find out about the different cuisines in each country. Present this data every week with at least 10 bullet points with charts and graphs to support the data.
                Also include photos from to be included in the blogs.
    """,

    agent=researcher,
)

task2 = Task(
    description="""Analyze the data from the researcher and the SEO specialist. Also, use your personal experience in travel to publish two unque articles per week on ecotourism for the blog jamesburnsmedia.com.
                Be sure to include proper Title and heading formats to make the blog look neat and be correct for Google to crawl and rank the site high. Include photos, dividers, and no more than 3 sentences 
                per paragraph so the visitor can read easily. Articles should be at least 1000 words.
    """,
    agent=travel_writer,
)

task3 = Task(
    description="""Analyze the SEO in each of the writer's blog posts for jamesburnsmedia.com. The website uses Kadence theme and Kadence blocks. Be sure SEO best practices are being used. Utilize the Rank Math plugin on jamesburnsmedia.com,
                create tags, and get the SEO score up to at least 90/100.
    """,
    agent=seo_specialist,
)

crew = Crew(
    agents=[researcher, travel_writer, seo_specialist],
    tasks=[task1, task2,task3],
    verbose=2,
    process=Process.sequential, # Sequential process will have tasks executed one after the other and the results of the previous one is passed as addtional content into the next task
)

result = crew.kickoff()

print("######################")
print(result)
