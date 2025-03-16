// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

client<llm> CustomGPT4o {
  provider openai
  options {
    model "gpt-4o"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> GPTo3mini {
  provider openai
  options {
    model "o3-mini"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> GPT45Preview {
  provider openai
  options {
    model "gpt-4.5-preview-2025-02-27"
    api_key env.OPENAI_API_KEY
    max_tokens 16384
  }
}

client<llm> CustomGPT4oMini {
  provider openai
  retry_policy Exponential
  options {
    model "gpt-4o-mini"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> CustomSonnet {
  provider anthropic
  options {
    model "claude-3-7-sonnet-20250219"
    api_key env.ANTHROPIC_API_KEY
  }
}

client<llm> CustomSonnetThinking {
  provider anthropic
  retry_policy Exponential
  options {
    model "claude-3-7-sonnet-20250219"
    api_key env.ANTHROPIC_API_KEY
    max_tokens 64000
    thinking {
      type enabled
      budget_tokens 16384
    }
  }
}


client<llm> CustomHaiku {
  provider anthropic
  retry_policy Constant
  options {
    model "claude-3-haiku-20240307"
    api_key env.ANTHROPIC_API_KEY
  }
}

client<llm> OpenRouter {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "google/gemma-3-27b-it"
  }
}

client<llm> Gemini2Flash {
  provider google-ai
  options {
    model "gemini-2.0-flash-thinking-exp-01-21"
    api_key env.GEMINI_API_KEY
  }
}


client<llm> Llama3 {
  // See https://docs.boundaryml.com/docs/snippets/clients/providers/ollama
  // to learn more about how to configure this client
  //
  // Note that you should run ollama using `OLLAMA_ORIGINS='*' ollama serve`
  // and you'll also need to `ollama pull llama3` to use this client
  provider ollama
  options {
    base_url "http://localhost:11434/v1"
    model "llama3.3:70b-instruct-q4_K_M"
    max_tokens 64000
    temperature 0.6
  }
}

client<llm> QwQ {
  // See https://docs.boundaryml.com/docs/snippets/clients/providers/ollama
  // to learn more about how to configure this client
  //
  // Note that you should run ollama using `OLLAMA_ORIGINS='*' ollama serve`
  // and you'll also need to `ollama pull llama3` to use this client
  provider ollama
  options {
    base_url "http://localhost:11434/v1"
    model "qwq"
  }
}
// https://docs.boundaryml.com/docs/snippets/clients/round-robin
client<llm> CustomFast {
  provider round-robin
  options {
    // This will alternate between the two clients
    strategy [CustomGPT4oMini, CustomHaiku]
  }
}

// https://docs.boundaryml.com/docs/snippets/clients/fallback
client<llm> OpenaiFallback {
  provider fallback
  options {
    // This will try the clients in order until one succeeds
    strategy [CustomGPT4oMini, CustomGPT4oMini]
  }
}

// https://docs.boundaryml.com/docs/snippets/clients/retry
retry_policy Constant {
  max_retries 3
  // Strategy is optional
  strategy {
    type constant_delay
    delay_ms 200
  }
}

retry_policy Exponential {
  max_retries 2
  // Strategy is optional
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}