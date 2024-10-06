import asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import backoff
import inspect
import boto3
import os
import json
from botocore.config import Config

class ReviewModelError(Exception):
    """Custom exception for review model errors."""
    pass

class PromptEvaluator:
    
    def __init__(self, df_train, target_model_name, target_model_native_version, target_model_config, review_model_name, review_model_config, review_prompt_template_path):
        #frame = inspect.currentframe()
        #args, _, _, values = inspect.getargvalues(frame)
        #print("Function parameters:")
        #for arg in args:
        #    print(f"{arg} = {values[arg]}")
        #increase the standard time out limits in boto3, because Bedrock may take a while to respond to large requests.
        my_config = Config(
            connect_timeout=60*3,
            read_timeout=60*3,
        )
        self.bedrock = boto3.client(service_name='bedrock-runtime',config=my_config)
        self.bedrock_service = boto3.client(service_name='bedrock',config=my_config)
        self.accept = 'application/json'
        self.contentType = 'application/json'
        
        self.df_train = df_train
        self.target_model_name = target_model_name
        self.target_model_native_version = target_model_native_version
        self.review_model_native_version = target_model_native_version
        self.target_model_config = target_model_config
        self.review_model_name = review_model_name
        self.review_model_config = review_model_config
        self.review_prompt_template_path = review_prompt_template_path


        
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def generate_target_model_response(self, question, prompt):
        #target_model = GenerativeModel(
        #   self.target_model_name,
        #   generation_config=self.target_model_config,
        #   system_instruction=prompt
        #)
        #print ("in generate_target_model_response")
        #print(self.target_model_native_version)
        # print("Local variables:")
        # current_frame = inspect.currentframe()
        # local_vars = current_frame.f_locals
        # for var_name, var_value in local_vars.items():
        #  print(f"{var_name}: {var_value}")
        #print("Printing body 1")
        body = json.dumps({
            "anthropic_version": self.target_model_native_version,
            "max_tokens": self.target_model_config['max_output_tokens'],
            "system": prompt,
            "temperature": self.target_model_config['temperature'],
            "top_p": self.target_model_config['top_p'],
            "messages": [
                {
                "role": "user",
                "content": [{"type": "text", "text": question}]
                }
            ],
        })
        #print("Printing body2")
        #print("Target: invoking bedrock body")
        response = self.bedrock.invoke_model(
        body=body, modelId=self.target_model_name, accept=self.accept, contentType=self.contentType
        )
        #print("invoking bedrock body")
        response_body = json.loads(response.get("body").read())
        #print(response_body)
        #outputText = response_body.get("content")[0].get("text")
        outputText = response_body["content"][0]["text"]
        # response = await target_model.generate_content_async(
        #    question,
        #     stream=False,
        # )                
        return outputText.strip().lower()

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def generate_review_model_response(self, review_prompt):
        #print ("in generate_review_model_response")
        body = json.dumps({
            "anthropic_version": self.review_model_native_version,
            "max_tokens": self.review_model_config['max_output_tokens'],
            "temperature": self.review_model_config['temperature'],
            "top_p": self.review_model_config['top_p'],
            "messages": [
                {
                "role": "user",
                "content": [{"type": "text", "text": review_prompt}]
                }
            ],
        })
        #print("review: invoking bedrock body")
        response = self.bedrock.invoke_model(
            body=body, modelId=self.review_model_name, accept=self.accept, contentType=self.contentType
        )
        response_body = json.loads(response.get("body").read())
        #print(response_body)
        outputText = response_body["content"][0]["text"]
        return outputText.strip().lower()

    async def generate_and_review(self, row, prompt):
        try:
            model_response = await self.generate_target_model_response(row["question"], prompt)

            # Load the review prompt from the text file
            with open(self.review_prompt_template_path, 'r') as f:
                review_prompt_template = f.read().strip()

            # Fill in the review prompt with the model response and ground truth
            review_prompt = review_prompt_template.format(model_response=model_response, ground_truth=row['answer'])

            # Now use the review model to compare the model response with the ground truth
            review_result = await self.generate_review_model_response(review_prompt)

            # Check if the target model returned a valid response
            if not model_response or not isinstance(model_response, str):
                raise ReviewModelError("Target model did not return a valid response.")

            # Assert that the review model returns either 'true' or 'false'
            if review_result not in ['true', 'false']:
                raise ReviewModelError("Review model did not return a valid response.")

            is_correct = review_result == 'true'  # Check if the response is 'True'

            return row.name, model_response, is_correct 
        except ReviewModelError as e:
            print(f"Error: {e}. The review model did not return a valid response. Terminating the program.")
            raise  # Re-raise the exception to be caught in the main function
        except Exception as e:
            print(f"An error occurred: {e}. Terminating the program.")
            raise  # Re-raise the exception to be caught in the main function

    async def evaluate_prompt(self, prompt):
        tasks = [self.generate_and_review(row, prompt) for _, row in self.df_train.iterrows()]

        # Create a tqdm progress bar
        with tqdm_asyncio(total=len(tasks), desc="Evaluating Prompt") as pbar:

            async def wrapped_task(task):
                result = await task
                pbar.update(1)  # Update progress bar after task completion
                return result

            # Run tasks with progress bar updates
            results = await asyncio.gather(*[wrapped_task(task) for task in tasks])

        # Prepare results for saving
        evaluation_results = []
        for index, model_response, is_correct in results:
            if index is not None:  # Check if the index is valid
                self.df_train.loc[index, 'model_response'] = model_response
                self.df_train.loc[index, 'is_correct'] = is_correct
                evaluation_results.append({
                    'question': self.df_train.loc[index, 'question'],
                    'ground_truth': self.df_train.loc[index, 'answer'],
                    'model_response': model_response,
                    'is_correct': is_correct
                })

        overall_accuracy = sum(self.df_train["is_correct"]) / len(self.df_train)

        # Save results to CSV
        results_df = pd.DataFrame(evaluation_results)
        results_csv_path = 'evaluation_results.csv'
        results_df.to_csv(results_csv_path, index=False)

        return overall_accuracy

    async def main(self, prompt):
        try:
            accuracy = await self.evaluate_prompt(prompt)
            print(f"Overall accuracy for the prompt: {accuracy:.2f}")
        except ReviewModelError:
            print("The program has terminated due to an invalid response from the review model.")
        except Exception as e:
            print(f"The program has terminated due to an unexpected error: {e}")