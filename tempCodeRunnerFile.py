from Frontend.GUI import (
    GraphicalUserInterface,
    SetAssistantStatus,
    ShowTextToScreen,
    GraphicsDirectoryPath,
    SetMicrophoneStatus,
    AnswerModifier,
    GetMicrophoneStatus,
    GetAssistantStatus,
    QueryModifier
)
from Backend.Model import FirstLayerDMM
from Backend.Chatbot import Chatbot
from Backend.Realtime import RealTimeSearchEngine
from Backend.SpeechToText import SpeechRecognition
from Backend.TextToSpeech import TextToSpeech
from Backend.Automation import Automation
from dotenv import dotenv_values
from asyncio import run
from time import sleep
import os
import subprocess
import threading
import json

env_vars = dotenv_values(".env")
Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
DefaultMessage=f'''{Username}:Hello {Assistantname}, How are you?
{Assistantname}:Welcome,{Username}. I am good, how can I help you?'''
subprocess=[]
Functions=["open","close","play","system","content","youtube search","reminder","google search"]

def ShowDefaultChatIfNoChats():
    File=open(r'Data\ChatLog.json','r',encoding='utf-8')
    if len(File.read())<5:
        with open(GraphicsDirectoryPath('Database.data'), 'w', encoding='utf-8') as file:
            file.write("")

        with open(GraphicsDirectoryPath('Responses.data'), 'w', encoding='utf-8') as file:
            file.write(DefaultMessage)

def ReadChatLogJson():
    with open(r'Data\ChatLog.json','r',encoding='utf-8') as file:
        chatlog_data=json.load(file)
    return chatlog_data

def ChatLogIntegration():
    json_data=ReadChatLogJson()
    formatted_chatlog=""
    for entry in json_data:
        if entry['role']=="User":
            formatted_chatlog+=f"{Username}:{entry['content']}\n"
        elif entry['role']=="Assistant":
            formatted_chatlog+=f"{Assistantname}:{entry['content']}\n"
    formatted_chatlog=formatted_chatlog.replace("User",Username + " ")
    formatted_chatlog=formatted_chatlog.replace("Assistant",Assistantname + " ")

    with open(GraphicsDirectoryPath('Database.data'), 'w', encoding='utf-8') as file:
        file.write(AnswerModifier(formatted_chatlog))


def ShowChatsOnGUI():
    File=open(GraphicsDirectoryPath('Database.data'),'r',encoding='utf-8')
    Data=File.read()
    if len(str(Data))>0:
        lines=Data.split("\n")
        result='\n'.join(lines)
        File.close()
        File=open(GraphicsDirectoryPath('Responses.data'),'w',encoding='utf-8')
        File.write(result)
        File.close()


def InitialExecution():
    SetMicrophoneStatus("False")
    ShowTextToScreen("")
    ShowDefaultChatIfNoChats()
    ChatLogIntegration()
    ShowChatsOnGUI()

InitialExecution()

def MainExecution():
    TaskExecution=False
    ImageExecution=False
    ImageGenerationQuery=""
    SetAssistantStatus("Listening.....")
    Query=SpeechRecognition()
    ShowTextToScreen(f"{Username}:{Query}")
    SetAssistantStatus("Thinking.....")
    Decision=FirstLayerDMM(Query)
    print("")
    print(f"Decision: {Decision}")
    print("")

    G=any([i for i in Decision if i.startswith("general")])
    R=any([i for i in Decision if i.startswith("realtime")])

    Mearged_query=" and ".join([" ".join(i.split()[1:]) for i in Decision if i.startswith("general") or i.startswith("realtime")])

    for queries in Decision:
        if "generate " in queries:
            ImageGenerationQuery=str(queries)
            ImageExecution=True

    for queries in Decision:
        if TaskExecution==False:
            if any(queries.startswith(func) for func in Functions):
                run(Automation(list(Decision)))
                TaskExecution=True

    if ImageExecution==True:

        with open(r"Frontend\Files\ImageGeneration.data","w") as file:
            file.write(f"{ImageGenerationQuery},True")

        try:
            p1=subprocess.Popen(["python",r"Backend\ImageGeneration.py"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                stdin=subprocess.PIPE,
                                shell=False)
            subprocess.append(p1)

        except Exception as e:
            print(f"Error in Image Generation: {e}")

    if G and R or R:
        
        SetAssistantStatus("Searching.....")
        Answer=RealTimeSearchEngine(QueryModifier(Mearged_query))
        ShowTextToScreen(f"{Assistantname}:{Answer}")
        SetAssistantStatus("Speaking.....")
        TextToSpeech(Answer)
        return True
    else:
        for Queries in Decision:
            if "general" in Queries:
                SetAssistantStatus("Thinking.....")
                QueryFinal=Queries.replace("general","")
                Answer=Chatbot(QueryModifier(QueryFinal))
                ShowTextToScreen(f"{Assistantname}:{Answer}")
                SetAssistantStatus("Answering.....")
                TextToSpeech(Answer)
                return True
            elif "realtime" in Queries:
                SetAssistantStatus("Searching.....")
                QueryFinal=Queries.replace("realtime","")
                Answer=RealTimeSearchEngine(QueryModifier(QueryFinal))
                ShowTextToScreen(f"{Assistantname}:{Answer}")
                SetAssistantStatus("Answering.....")
                TextToSpeech(Answer)
                return True
            elif "exit" in Queries:
                QueryFinal="Okay,Bye"
                Answer=Chatbot(QueryModifier(QueryFinal))
                ShowTextToScreen(f"{Assistantname}:{Answer}")   
                SetAssistantStatus("Answering.....")
                TextToSpeech(Answer)
                SetAssistantStatus("Answering.....")
                os._exit(1)

def FirstThread():
    while True:
        CurrentStatus=GetMicrophoneStatus()
        if CurrentStatus=="True":
            MainExecution()

        else:
            AIStatus=GetAssistantStatus()

            if "Available..." in AIStatus:
                sleep(0.1)

            else:
                SetAssistantStatus("Available...")
                
def SecondThread():
    GraphicalUserInterface()

if __name__=="__main__":
    thread2=threading.Thread(target=FirstThread,daemon=True)
    thread2.start()
    SecondThread()