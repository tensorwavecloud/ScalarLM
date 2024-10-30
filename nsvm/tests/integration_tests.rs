// importing common module.

use std::io;
use openai_api_rust::OpenAI;
use openai_api_rust::*;
use openai_api_rust::chat::*;

#[test]
fn test_add() {
    // using common code.
    assert_eq!(3 + 2, 5);
}

#[test]
fn test_openai() {
    // Load API key from environment OPENAI_API_KEY.
    // You can also hadcode through `Auth::new(<your_api_key>)`, but it is not recommended.
    let auth = Auth::from_env().unwrap();
    let openai = OpenAI::new(auth, "https://api.openai.com/v1/");
    let body = ChatBody {
        model: "gpt-3.5-turbo".to_string(),
        max_tokens: Some(7),
        temperature: Some(0_f32),
        top_p: Some(0_f32),
        n: Some(2),
        stream: Some(false),
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        user: None,
        messages: vec![Message { role: Role::User, content: "Hello!".to_string() }],
    };
    let rs = openai.chat_completion_create(&body);
    let choice = rs.unwrap().choices;
    let message = &choice[0].message.as_ref().unwrap();
    assert!(message.content.contains("Hello"));
}

#[test]
fn test_kuzu() ->  Result<(), Box<dyn std::error::Error>> {
    use kuzu::{Database, SystemConfig, Connection};

    let db = Database::new("./tmp.db", SystemConfig::default())?;
    let conn = Connection::new(&db)?;
    conn.query("CREATE NODE TABLE Person(name STRING, age INT64, PRIMARY KEY(name));")?;
    conn.query("CREATE (:Person {name: 'Alice', age: 25});")?;
    conn.query("CREATE (:Person {name: 'Bob', age: 30});")?;

    let mut result = conn.query("MATCH (a:Person) RETURN a.name AS NAME, a.age AS AGE;")?;
    println!("{}", result.display());
    Ok(())
}