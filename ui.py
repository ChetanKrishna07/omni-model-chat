import gradio as gr
from completion import run_task_omni, run_task_solo



def greet(Chat, model):
    task = {"Question": Chat, "Answer": "0.0"}
    if model == "omni_model":
        result = run_task_omni(task)
    else:
        result = run_task_solo(task)
        
    correct, cost, solution_dict = result
    answer = solution_dict["Answer"]
    print(solution_dict)
    return answer, f"${cost}", solution_dict["Question"]

demo = gr.Interface(
    fn=greet, 
    inputs=[
        "text",
        gr.Radio(["omni_model", "solo_model"]),    
    ],
    outputs=["text", "text", "text"])

demo.launch()