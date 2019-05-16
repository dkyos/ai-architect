curl -i -v -X POST \
    -H "Content-Type:application/json" \
    -H "RESPONSE-FORMAT:application/json" \
    http://localhost:8080/inference \
    -d '{"model_name": "word2vec", "docs": [{"id": 1, "doc": "Mars"}]}' | jq .
