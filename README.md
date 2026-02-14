# NCAA Men's Basketball Deep Learning Betting Model

A sophisticated deep learning model that predicts NCAA men's basketball game outcomes and provides betting insights including spread analysis and over/under recommendations.

## Features

- **Deep Neural Network**: Multi-layer neural network with dropout and batch normalization
- **Comprehensive Statistics**: Uses 30+ features including:
  - Basic stats (PPG, FG%, 3P%, FT%, rebounds, assists, turnovers, steals, blocks)
  - Advanced metrics (offensive/defensive rating, pace, win percentages)
  - Recent form and home/away splits
- **Betting Insights**: 
  - Win probability predictions
  - Point spread value analysis
  - Over/under recommendations
  - Confidence ratings
- **Interactive Input**: User-friendly prompts for game data entry

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
```bash
pip install numpy pandas scikit-learn
```

Note: This version uses scikit-learn's MLPClassifier instead of TensorFlow for better compatibility.

## Usage

Run the model:
```bash
python ncaa_betting_model_sklearn.py
```

Or run the demo first to see it in action:
```bash
python demo.py
```

The program will:
1. Train the neural network (takes ~30 seconds)
2. Prompt you for team statistics
3. Generate betting insights and recommendations
4. Allow you to analyze multiple games

## Input Requirements

You'll be asked to provide:

### Team Statistics
- Points per game
- Field goal percentage
- 3-point percentage
- Free throw percentage
- Rebounds per game
- Assists per game
- Turnovers per game
- Steals per game
- Blocks per game
- Overall win percentage
- Home/away win percentage
- Recent form (wins in last 5 games)

### Game Details
- Home/away designation
- Current point spread
- Over/under line (optional)

### Advanced Stats (Optional)
- Pace (possessions per game)
- Offensive rating
- Defensive rating

If you don't have advanced stats, the model uses reasonable defaults.

## Where to Find Statistics

You can find NCAA basketball statistics on:
- **ESPN.com**: Team stats pages
- **Sports-Reference.com**: Comprehensive historical data
- **KenPom.com**: Advanced metrics (pace, offensive/defensive ratings)
- **TeamRankings.com**: Detailed team statistics
- **NCAA.com**: Official statistics

## Output Interpretation

The model provides:

1. **Win Probability**: Likelihood of the team winning (0-100%)
2. **Predicted Margin**: Expected point differential
3. **Spread Analysis**: 
   - Value bets are flagged when model disagrees with the spread
   - Shows percentage edge
4. **Over/Under Analysis**: Predicted total vs. line
5. **Confidence Level**: How certain the model is about the prediction

## Example Session

```
Enter your team name: Duke
Enter opponent team name: North Carolina

--- Duke Statistics ---
Points per game: 82.5
Field goal % (e.g., 45.5 for 45.5%): 47.2
3-point % (e.g., 35.0 for 35%): 36.8
...

============================================================
🏀 BETTING INSIGHTS: Duke vs North Carolina
============================================================

📊 Win Probability: 67.3%
📊 Loss Probability: 32.7%
📈 Predicted Margin: +3.5 points

💰 SPREAD ANALYSIS (Current Spread: -2.5)
   ✓ VALUE BET: Duke (8.2% edge)
   Recommendation: Bet Duke -2.5

⭐ Confidence Level: 68.6%
   High Confidence
============================================================
```

## Model Architecture

- Input Layer: 30 features
- Hidden Layer 1: 128 neurons (ReLU, Dropout 0.3)
- Hidden Layer 2: 64 neurons (ReLU, Batch Norm, Dropout 0.2)
- Hidden Layer 3: 32 neurons (ReLU, Dropout 0.2)
- Hidden Layer 4: 16 neurons (ReLU)
- Output Layer: 1 neuron (Sigmoid activation for probability)

Training uses:
- Adam optimizer
- Binary cross-entropy loss
- Early stopping to prevent overfitting

## Important Notes

⚠️ **Disclaimer**: This model is for educational and entertainment purposes only. Always bet responsibly and within your means. Past performance does not guarantee future results.

### Limitations
- Model is trained on synthetic data representative of NCAA basketball
- For best results with real games, retrain on actual historical game data
- Does not account for: injuries, coaching changes, weather, momentum, or intangibles
- Betting lines reflect more information than any model can capture

### Tips for Better Predictions
1. Use the most recent statistics available
2. Account for recent injuries or lineup changes manually
3. Consider the importance of the game (tournament vs. regular season)
4. Look for patterns across multiple similar matchups
5. Use the model as one input among many in your analysis

## Customization

You can modify the model by:
- Adjusting network architecture in `build_model()`
- Adding more features to `feature_names`
- Training on real historical data instead of synthetic data
- Tuning hyperparameters (learning rate, dropout rates, etc.)

## Future Enhancements

Potential improvements:
- Integration with live statistics APIs
- Historical game database
- Player-level analysis
- Ensemble methods combining multiple models
- Situational factors (rest days, travel, etc.)
- Conference-specific models

## License

Free to use and modify for personal, educational, and non-commercial purposes.

## Support

For issues or questions, ensure you have the latest versions of all dependencies installed.

**Remember: Gamble responsibly. Never bet more than you can afford to lose.**
