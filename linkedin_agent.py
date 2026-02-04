from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import os

from dotenv import load_dotenv

load_dotenv()

print(os.getenv("OPENROUTER_API_KEY"))

class LinkedinMaker:
    def __init__(self,topic,model,temperature=None):

        self.topic=topic
        self.model=model
        self._temperature=temperature
        self._strparser=StrOutputParser()
        
    def _makeAgent(self):
        print(os.getenv("OPENROUTER_API_KEY"))
        return init_chat_model(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=self.model,
            temperature=self._temperature
        )
    def _makePrompt(self,human,system):
        return ChatPromptTemplate.from_messages([
            ('system',system),
            ('user',human)
        ])
    
    def _dictMaker(self):
        return RunnableLambda(lambda x: {"content":x})

    def getOutput(self):
        prompt_t1=self._makePrompt(
            human="write facts about {topic}",
            system="You are an expert in this {topic}"
        )
        prompt_t2=self._makePrompt(
            human="write a linkedin post on {content}",
            system="You are a great linkedin content creator"
        )
        dict_maker_runnable=self._dictMaker()
        agent=self._makeAgent()
        pipe= prompt_t1|agent|self._strparser|dict_maker_runnable|prompt_t2|agent|self._strparser
        output=pipe.invoke(self.topic)


MainAgent=LinkedinMaker("Organoid Intelligence","gpt-oss-120b:free")
print(MainAgent.getOutput())