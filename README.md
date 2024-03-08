# ai-egn-poc


## Quickstart

1. Install requirements
   ```
   $ pip install -r requirements.txt
   ```

2. Setup API key

   You should export your openai-api-key as an environment variable as shown below.
   ```
   $ export OPENAI_API_KEY='your-api-key-here'
   ```

3. Run app
   ```
   $ uvicorn main:app --reload
   ```
   or
   ```
   $ python -m uvicorn main:app --reload
   ```
   