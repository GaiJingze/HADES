#!/bin/bash

API_KEY="your_openai_api_key_here"
SCENARIO_FILE="./benchmark/scenario.json"
MODEL="gpt-4-0314" 

# Array of scenarios
SCENARIOS=('Animal' 'Self-Harm' 'Privacy' 'Violence' 'Financial')

for SCENARIO in "${SCENARIOS[@]}"; do
  echo "Processing scenario: $SCENARIO"
  
  # Generate keywords
  python benchmark/generate_keywords.py \
    --api_key "$API_KEY" \
    --scenario "$SCENARIO" \
    --scenario_file "$SCENARIO_FILE"
  
  
  # Generate instructions
  python benchmark/generate_instructions.py \
    --api_key "$API_KEY" \
    --scenario "$SCENARIO" \
    --scenario_file "$SCENARIO_FILE"
  
  echo "Finished processing scenario: $SCENARIO"
done

echo "All scenarios processed successfully."
