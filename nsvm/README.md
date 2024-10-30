# Neuro Symbolic Virtual Machine


json schema to Rust codegen to json schema
json schema to cypher CREATE
json object to cypher INSERT

JsonSchemas
Db interface
OpenAI sdk

1. create schemas folder
    specification.json
    function.json
    test_case.json

2. main fn that accepts a specification
   * check spec
   * stores in db

3. map a json object and schema into the db

AddSchema(schema) -- creates the db schema
Store(object, schema)



2. end-to-end test for text-to-sql


# Papers

ExeDec: Execution Decomposition for Compositional Generalization in Neural Program Synthesis
https://github.com/google-deepmind/exedec

Relational Decomposition for Program Synthesis
https://arxiv.org/pdf/2408.12212