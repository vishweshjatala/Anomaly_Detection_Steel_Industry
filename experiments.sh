#!/bin/bash

# Activate your Python environment if needed
# source /path/to/your/venv/bin/activate

# Set the necessary arguments
EPOCHS=100  # Default number of epochs
LR=0.001  # Learning rate
BS=128  # Batch size
PATIENCE=15  # Early stopping patience
SEQ_LEN=30

# Define regions, models, and profiles
REGIONS=("Roughing" "Intermediate" "Finishing" "FFB1" "FFB2")
MODELS=("pca" "lstm" "usad")  # Other models like "pca" and "lstm" can be added if needed
CUSTOM_SCALERS=(true false)
PROFILES=("12mm")
THRESHOLD="z-score"

# Loop through each profile, region, and model type
for PROFILE in "${PROFILES[@]}"; do
    for REGION in "${REGIONS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            # Set SEQ_LEN and THRESHOLD based on the model type
            if [ "$MODEL" = "usad" ]; then
                SEQ_LEN=10
                THRESHOLD="pct"
            elif [ "$MODEL" = "tcn" ]; then
                EPOCHS=50
                PATIENCE=5
            else
                SEQ_LEN=30  # Default sequence length for other models
                THRESHOLD="z-score"
            fi

            for CUSTOM_SCALER in "${CUSTOM_SCALERS[@]}"; do
                # Construct the training command
                CMD_TRAIN="python trainer.py --profile $PROFILE --region $REGION --model $MODEL --epochs $EPOCHS --lr $LR --seq_len $SEQ_LEN --bs $BS --patience $PATIENCE"
                CMD_TUNE="python tuner.py --profile $PROFILE --region $REGION --model $MODEL --threshold $THRESHOLD"
                CMD_CONTINUAL="python continual.py --profile $PROFILE --region $REGION --model $MODEL --epochs 50 --lr $LR --seq_len $SEQ_LEN --bs 16"

                # Add custom_scaler flag if needed
                if [ "$CUSTOM_SCALER" = true ]; then
                    CMD_TRAIN="$CMD_TRAIN --custom_scaler"
                    CMD_TUNE="$CMD_TUNE --custom_scaler"
                    CMD_CONTINUAL="$CMD_CONTINUAL --custom_scaler"
                fi

                # Print and execute the commands
                echo "Running training for PROFILE: $PROFILE, REGION: $REGION, MODEL: $MODEL, CUSTOM_SCALER: $CUSTOM_SCALER, SEQ_LEN: $SEQ_LEN"
                echo "Executing: $CMD_TRAIN"
                eval $CMD_TRAIN
                echo "Running tuning for PROFILE: $PROFILE, REGION: $REGION, MODEL: $MODEL, CUSTOM_SCALER: $CUSTOM_SCALER, SEQ_LEN: $SEQ_LEN"
                echo "Executing: $CMD_TUNE"
                eval $CMD_TUNE
                echo "Running continual-learning for PROFILE: $PROFILE, REGION: $REGION, MODEL: $MODEL, CUSTOM_SCALER: $CUSTOM_SCALER, SEQ_LEN: $SEQ_LEN"
                echo "Executing: $CMD_CONTINUAL"
                eval $CMD_CONTINUAL
            done
        done
    done
done