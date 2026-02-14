# Quick Start Guide

## Get Started in 3 Steps

### 1. Install Dependencies
```bash
pip install numpy pandas scikit-learn
```

### 2. Run the Demo (Recommended First)
```bash
python demo.py
```
This will show you how the model works with a sample Duke vs UNC game.

### 3. Analyze Your Own Games
```bash
python ncaa_betting_model_sklearn.py
```
Follow the prompts to enter team statistics and get betting insights.

## Sample Input Session

When you run the program, you'll be prompted like this:

```
Enter your team name: Duke
Enter opponent team name: UNC

--- Duke Statistics ---
Points per game: 82.5
Field goal % (e.g., 45.5 for 45.5%): 47.2
3-point % (e.g., 35.0 for 35%): 36.8
Free throw % (e.g., 72.0 for 72%): 75.4
Rebounds per game: 38.2
Assists per game: 16.3
Turnovers per game: 11.8
Steals per game: 7.5
Blocks per game: 4.8
Win % (e.g., 75.0 for 75%): 78.0
Home win % (e.g., 85.0 for 85%): 88.0
Recent form - wins in last 5 (0-5): 4
```

...and the same for the opponent.

## Understanding the Output

The model gives you:

1. **Win Probability** - Chance the team wins (0-100%)
2. **Predicted Margin** - Expected point differential
3. **Spread Analysis** - Whether to bet the favorite or underdog
4. **Over/Under Analysis** - Whether to bet over or under the total
5. **Confidence Level** - How certain the model is

## Where to Find Stats

- **ESPN.com** - Basic stats (PPG, FG%, rebounds, etc.)
- **KenPom.com** - Advanced stats (offensive/defensive ratings, pace)
- **Sports-Reference.com** - Historical data and season averages
- **TeamRankings.com** - Comprehensive team statistics

## Tips

1. Use season averages, not single-game stats
2. Recent form (last 5 games) is important for hot/cold teams
3. Home/away splits matter significantly
4. Advanced stats (if available) improve accuracy
5. Use the model as one input, not the only factor

## Disclaimer

⚠️ This model is for educational purposes only. Bet responsibly and never wager more than you can afford to lose.
