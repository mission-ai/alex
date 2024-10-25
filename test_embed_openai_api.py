from openai import OpenAI

client = OpenAI(
  api_key="$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
  base_url="https://pubnv-embedqa-nim.ufhpc/v1"
)

response = client.embeddings.create(
    input=["What is the capital of France?"],
    model="nvidia/nv-embedqa-e5-v5",
    encoding_format="float",
    extra_body={"input_type": "query", "truncate": "NONE"}
)

print(response.data[0].embedding)