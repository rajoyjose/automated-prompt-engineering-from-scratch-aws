import asyncio
import pandas as pd
from prompt_evaluator import PromptEvaluator
#from vertexai.generative_models import HarmBlockThreshold, HarmCategory
#imports for AWS 
import boto3
import os

region = os.environ.get("AWS_REGION")
bedrock_service = boto3.client(
    service_name='bedrock',
    region_name=region,
)

if __name__ == "__main__":
    df_train = pd.read_csv('test.csv')  # Load your training data
# Model ids: 
# Anthropic	Claude 3 Sonnet	1.0	anthropic.claude-3-sonnet-20240229-v1:0
# Anthropic	Claude 3.5 Sonnet	1.0	anthropic.claude-3-5-sonnet-20240620-v1:0
# Anthropic	Claude 3 Haiku	1.0	anthropic.claude-3-haiku-20240307-v1:0
# Anthropic	Claude 3 Opus	1.0	anthropic.claude-3-opus-20240229-v1:0

    target_model_name = "anthropic.claude-3-haiku-20240307-v1:0"
    #target_model_name = "anthropic.claude-3-sonnet-20240229-v1:0"
    target_model_native_version = "bedrock-2023-05-31"
    target_model_config = {
        "temperature": 0, "max_output_tokens": 1000, "top_p":0
        }
    review_model_name = "anthropic.claude-3-haiku-20240307-v1:0"
    review_model_config = {
        "temperature": 0, "max_output_tokens": 10, "top_p":0
        }
    review_prompt_template_path = 'review_prompt_template.txt'
    evaluator = PromptEvaluator(df_train, target_model_name, target_model_native_version, target_model_config, review_model_name, review_model_config, review_prompt_template_path)
    
    prompt = input("Please enter the prompt for evaluation: ")
    asyncio.run(evaluator.main(prompt))