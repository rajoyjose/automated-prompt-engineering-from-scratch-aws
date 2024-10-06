import asyncio
import os
import pandas as pd
#from vertexai.generative_models import GenerativeModel, HarmBlockThreshold, HarmCategory
import re
import aiofiles
import datetime
import aioconsole
from prompt_evaluator import PromptEvaluator
import backoff
import json

#imports for AWS 
import boto3
import os
from botocore.config import Config

region = os.environ.get("AWS_REGION")
bedrock_service = boto3.client(
    service_name='bedrock',
    region_name=region,
)

class APD:
    def __init__(self, num_prompts, starting_prompt, df_train, metaprompt_template_path, generation_model_name, generation_model_config, target_model_name, target_model_native_version, target_model_config, review_model_name, review_model_config, review_prompt_template_path):
        self.num_prompts = num_prompts
        self.starting_prompt = starting_prompt
        self.df_train = df_train
        self.metaprompt_template_path = metaprompt_template_path
        self.generation_model_name = generation_model_name
        self.generation_model_config = generation_model_config
        self.target_model_native_version = target_model_native_version
        #self.safety_settings = safety_settings
        # Bedrock client
        #increase the standard time out limits in boto3, because Bedrock may take a while to respond to large requests.
        my_config = Config(
            connect_timeout=60*3,
            read_timeout=60*3,
        )
        self.bedrock = boto3.client(service_name='bedrock-runtime',config=my_config)
        self.bedrock_service = boto3.client(service_name='bedrock',config=my_config)
        self.accept = 'application/json'
        self.contentType = 'application/json'
        # Initialize the generation model - GCP
        # self.generation_model = GenerativeModel(self.generation_model_name)
        # Create the "runs" folder if it doesn't exist
        self.runs_folder = "runs"
        os.makedirs(self.runs_folder, exist_ok=True)
        
        self.run_folder = self.create_run_folder()
        self.prompt_history = os.path.join(self.run_folder, 'prompt_history.txt')
        self.prompt_history_chronlogical = os.path.join(self.run_folder, 'prompt_history_chronlogical.txt')
        
        # Initialize the PromptEvaluator
        self.prompt_evaluator = PromptEvaluator(df_train, target_model_name, target_model_native_version, target_model_config, review_model_name, review_model_config, review_prompt_template_path)
        #self.prompt_evaluator = PromptEvaluator(
        #    df_train,
        #    target_model_name,
        #    target_model_config,
        #    review_model_name,
        #    review_model_config,
        #    safety_settings,
        #    review_prompt_template_path
        #)
        self.user_feedback = ""
        self.best_prompt = starting_prompt
        self.best_accuracy = 0.0

    def create_run_folder(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join(self.runs_folder, f'run_{timestamp}')  # Join with runs_folder
        os.makedirs(run_folder, exist_ok=True)
        return run_folder

    def create_prompt_subfolder(self, prompt_number):
        prompt_folder = os.path.join(self.run_folder, f'prompt_{prompt_number}')
        os.makedirs(prompt_folder, exist_ok=True)
        return prompt_folder

    def read_and_sort_prompt_accuracies(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        pattern = re.compile(r'<PROMPT>\n<PROMPT_TEXT>\n(.*?)\n</PROMPT_TEXT>\n<ACCURACY>\nAccuracy: ([0-9.]+)\n</ACCURACY>\n</PROMPT>', re.DOTALL)
        matches = pattern.findall(content)
        
        sorted_prompts = sorted(matches, key=lambda x: float(x[1]))  # Sort in ascending order
        return sorted_prompts

    def write_sorted_prompt_accuracies(self, file_path, sorted_prompts):
        sorted_prompts_string = ""
        with open(file_path, 'w') as f:
            for prompt, accuracy in sorted_prompts:
                s = f"<PROMPT>\n<PROMPT_TEXT>\n{prompt}\n</PROMPT_TEXT>\n<ACCURACY>\nAccuracy: {accuracy}\n</ACCURACY>\n</PROMPT>\n\n"
                f.write(s)
                sorted_prompts_string += s
                
        return sorted_prompts_string

    def update_metaprompt(self, file_path, metaprompt_template_path):
        sorted_prompts = self.read_and_sort_prompt_accuracies(file_path)
        sorted_prompts_string = self.write_sorted_prompt_accuracies(file_path, sorted_prompts)
                
        with open(metaprompt_template_path, 'r') as f:
            metaprompt_template = f.read()
        
        metaprompt = metaprompt_template.format(prompt_scores=sorted_prompts_string, human_feedback=self.user_feedback)
        return metaprompt

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def generate_with_backoff(self, metaprompt):
        #GCP Code 
        #response = self.generation_model.generate_content(
        #    metaprompt,
        #    generation_model_config=self.generation_model_config,
        #    stream=False,
        #)
        #print("generate: invoking bedrock body")
        # Generate the next prompt
        body = json.dumps({
            "anthropic_version": self.target_model_native_version,
            "max_tokens": self.generation_model_config['max_output_tokens'],
            "temperature": self.generation_model_config['temperature'],
            "top_p": self.generation_model_config['top_p'],
            "messages": [
                {
                "role": "user",
                "content": [{"type": "text", "text": metaprompt}]
                }
            ],
        })
        print("generate: invoking bedrock body")
        response = self.bedrock.invoke_model(
            body=body, modelId=self.generation_model_name, accept=self.accept, contentType=self.contentType
        )
        response_body = json.loads(response.get("body").read())
        print(response_body)
        outputText = response_body["content"][0]["text"]
        return outputText.strip().lower()

    async def main(self):
        prompt_accuracies = []

        for i in range(self.num_prompts + 1):
            await aioconsole.aprint("=" * 150)
            await aioconsole.aprint(f"Prompt number {i}")

            if i == 0:
                new_prompt = self.starting_prompt
                # Evaluate the starting prompt
                # accuracy = await self.prompt_evaluator.evaluate_prompt(new_prompt)
                # self.best_accuracy = accuracy
                # prompt_accuracies.append((new_prompt, accuracy))
            else:
                metaprompt = self.update_metaprompt(self.prompt_history, self.metaprompt_template_path)
                
                try:
                    response = await self.generate_with_backoff(metaprompt)
                except Exception as e:
                    await aioconsole.aprint(f"Failed to generate content after retries: {e}")
                    continue
                
                await aioconsole.aprint("-" * 150)
                await aioconsole.aprint(response)
                await aioconsole.aprint("-" * 150)
                
                match = re.search(r'\[\[(.*?)\]\]', response, re.DOTALL)
                if match:
                    new_prompt = match.group(1)
                else:
                    await aioconsole.aprint("No new prompt found")
                    continue
            
            # Create a subfolder for the prompt
            prompt_folder = self.create_prompt_subfolder(i)

            # Save the prompt in a text file within the subfolder
            prompt_file_path = os.path.join(prompt_folder, 'prompt.txt')
            with open(prompt_file_path, 'w') as f:
                f.write(new_prompt)

            # Use the PromptEvaluator to evaluate the new prompt
            accuracy = await self.prompt_evaluator.evaluate_prompt(new_prompt)
            
            if i == 0:
                best_accuracy = starting_accuracy = accuracy
            
            prompt_accuracies.append((new_prompt, accuracy))
            await aioconsole.aprint("-" * 150)
            await aioconsole.aprint(f"Overall accuracy for prompt: {accuracy:.2f}")
            await aioconsole.aprint("=" * 150)

            # Update the best prompt if the current accuracy is higher
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_prompt = new_prompt
            
            # Append to prompt_history.txt
            async with aiofiles.open(self.prompt_history, 'a') as f:
                await f.write(f"<PROMPT>\n<PROMPT_TEXT>\n{new_prompt}\n</PROMPT_TEXT>\n<ACCURACY>\nAccuracy: {accuracy:.2f}\n</ACCURACY>\n</PROMPT>\n\n")
        
            # Append to prompt_history_chronological.txt with prompt number
            async with aiofiles.open(self.prompt_history_chronlogical, 'a') as f:
                await f.write(f"Prompt number: {i}\nPrompt: {new_prompt}\nAccuracy: {accuracy:.2f}\n\n")
                await f.write("=" * 150 + "\n")
            
            # Save the evaluation results in a CSV file within the subfolder
            csv_file_path = os.path.join(prompt_folder, 'evaluation_results.csv')
            evaluation_results = {
                "question": self.df_train["question"],
                "answer": self.df_train["answer"],
                "model_response": self.df_train["model_response"],
                "is_correct": self.df_train["is_correct"]
            }
            evaluation_df = pd.DataFrame(evaluation_results)
            evaluation_df.to_csv(csv_file_path, index=False)

            # Read, sort, and write the updated prompt accuracies to prompt_history.txt
            sorted_prompts = self.read_and_sort_prompt_accuracies(self.prompt_history)
            self.write_sorted_prompt_accuracies(self.prompt_history, sorted_prompts)

        # Output the final best prompt and improvement in accuracy
        improvement = best_accuracy - starting_accuracy   # Compare to the last evaluated accuracy
        await aioconsole.aprint("=" * 150)
        await aioconsole.aprint(f"Final best prompt: {self.best_prompt}")
        await aioconsole.aprint(f"Accuracy of best prompt: {best_accuracy:.2f}")
        await aioconsole.aprint(f"Improvement in accuracy: {improvement:.2f}")

if __name__ == "__main__":
    num_prompts = 5
    #starting_prompt = "Solve the given problem about geometric shapes. Think step by step."
    starting_prompt = input("Please enter the prompt for evaluation: ")
    
    df_train = pd.read_csv('train.csv')  # Load your training data
    
    #GCP 
    #safety_settings = {
    #    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    #    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    #    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    #    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    #}  
   
    metaprompt_template_path = 'metaprompt_template.txt'
    generation_model_name = "anthropic.claude-3-sonnet-20240229-v1:0"
    #generation_model_name = "anthropic.claude-3-haiku-20240307-v1:0"
    generation_model_config = {
        "temperature": 0.7, "max_output_tokens": 1000, "top_p":0
        }
    target_model_name = "anthropic.claude-3-sonnet-20240229-v1:0"
    target_model_native_version = "bedrock-2023-05-31"
    target_model_config = {
        "temperature": 0, "max_output_tokens": 1000, "top_p":0
        }
    review_model_name = "anthropic.claude-3-sonnet-20240229-v1:0"
    review_model_config = {
        "temperature": 0, "max_output_tokens": 10, "top_p":0
        }
    review_prompt_template_path = 'review_prompt_template.txt'
    apd = APD(
        num_prompts, starting_prompt, df_train, 
        metaprompt_template_path, generation_model_name, generation_model_config,
        target_model_name, target_model_native_version, target_model_config, review_model_name, review_model_config, review_prompt_template_path
    )

    asyncio.run(apd.main())