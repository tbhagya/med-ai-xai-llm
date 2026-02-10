"""
LLM Wrapper Module

OpenAI-compatible LLM wrapper for flexible integration.
Supports any LLM service that implements OpenAI API format.
"""

from openai import OpenAI


class LLMWrapper:
    """
    OpenAI-compatible LLM wrapper for flexible integration.
    Supports any LLM service that implements OpenAI API format.
    """
    
    def __init__(self, base_url, api_key, model_name):
        """
        Initialize LLM client.
        
        Args:
            base_url (str): Base URL of the LLM service
            api_key (str): API key for authentication
            model_name (str): Model name/identifier
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
    
    def generate_completion(
        self,
        system_prompt,
        user_prompt
    ):
        """
        Generate text completion using chat API.
        
        Args:
            system_prompt (str): System message defining assistant behavior
            user_prompt (str): User query/request
            
        Returns:
            str: Generated text response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating completion: {str(e)}"
    
    def generate_completion_with_metadata(
        self,
        system_prompt,
        user_prompt,
        max_tokens=500,
        temperature=0.0,
        seed=42
    ):
        """
        Generate completion with full metadata.
        
        Args:
            system_prompt (str): System message defining assistant behavior
            user_prompt (str): User query/request
            max_tokens (int): Maximum tokens in response
            temperature (float): Sampling temperature (0.0 = deterministic)
            seed (int): Random seed for reproducibility
        
        Returns:
            dict: Contains 'text', 'usage', 'model', 'finish_reason'
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed
            )
            
            return {
                'text': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason
            }
        except Exception as e:
            return {
                'text': f"Error: {str(e)}",
                'usage': None,
                'model': None,
                'finish_reason': 'error'
            }