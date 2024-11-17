from dotenv import load_dotenv
from graph.graph import app

load_dotenv()

if __name__ =='__main__':
    # Prompt the user for input
    user_question = input("Enter your question: ")
    # Invoke the app with the user's input
    response = app.invoke(input={"question": user_question})
    # Print the response
    print(response["generation"])