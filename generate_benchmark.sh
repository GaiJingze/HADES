#!/bin/bash

API_KEY="aaaa"
SCENARIO_FILE="./benchmark/scenario.json"
MODEL="gpt-4-0314" 

# Array of scenarios
SCENARIOS=('Harassment' 'Hate' 'IllegalActivity' 'Self-Harm' 'Sexual' 'Violence')

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

   python benchmark/generate_category.py \
    --api_key "$API_KEY" \
    --scenario "$SCENARIO" \
    --model "$MODEL"
  
  echo "Finished processing scenario: $SCENARIO"
done

echo "All scenarios processed successfully."
