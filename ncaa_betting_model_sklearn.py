"""
NCAA Men's Basketball Deep Learning Betting Model
This model uses neural networks to predict game outcomes and betting insights
Uses scikit-learn's MLPClassifier for compatibility
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class NCAABettingModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'team_ppg', 'opp_ppg', 'team_fg_pct', 'opp_fg_pct',
            'team_3p_pct', 'opp_3p_pct', 'team_ft_pct', 'opp_ft_pct',
            'team_reb_pg', 'opp_reb_pg', 'team_ast_pg', 'opp_ast_pg',
            'team_to_pg', 'opp_to_pg', 'team_stl_pg', 'opp_stl_pg',
            'team_blk_pg', 'opp_blk_pg', 'team_win_pct', 'opp_win_pct',
            'team_home_win_pct', 'opp_away_win_pct', 'is_home',
            'team_pace', 'opp_pace', 'team_off_rating', 'opp_off_rating',
            'team_def_rating', 'opp_def_rating', 'team_recent_form', 'opp_recent_form'
        ]
    
    def generate_synthetic_data(self, n_samples=5000):
        """
        Generate synthetic training data based on realistic NCAA basketball statistics
        """
        np.random.seed(42)
        
        data = []
        for _ in range(n_samples):
            # Team statistics (realistic ranges)
            team_ppg = np.random.normal(75, 8)
            opp_ppg = np.random.normal(75, 8)
            team_fg_pct = np.random.normal(0.45, 0.05)
            opp_fg_pct = np.random.normal(0.45, 0.05)
            team_3p_pct = np.random.normal(0.35, 0.06)
            opp_3p_pct = np.random.normal(0.35, 0.06)
            team_ft_pct = np.random.normal(0.72, 0.06)
            opp_ft_pct = np.random.normal(0.72, 0.06)
            team_reb_pg = np.random.normal(37, 5)
            opp_reb_pg = np.random.normal(37, 5)
            team_ast_pg = np.random.normal(14, 3)
            opp_ast_pg = np.random.normal(14, 3)
            team_to_pg = np.random.normal(13, 3)
            opp_to_pg = np.random.normal(13, 3)
            team_stl_pg = np.random.normal(7, 2)
            opp_stl_pg = np.random.normal(7, 2)
            team_blk_pg = np.random.normal(4, 1.5)
            opp_blk_pg = np.random.normal(4, 1.5)
            team_win_pct = np.random.uniform(0.3, 0.9)
            opp_win_pct = np.random.uniform(0.3, 0.9)
            team_home_win_pct = team_win_pct + np.random.uniform(0, 0.15)
            opp_away_win_pct = opp_win_pct - np.random.uniform(0, 0.10)
            is_home = np.random.choice([0, 1])
            team_pace = np.random.normal(70, 5)
            opp_pace = np.random.normal(70, 5)
            team_off_rating = np.random.normal(110, 10)
            opp_off_rating = np.random.normal(110, 10)
            team_def_rating = np.random.normal(100, 10)
            opp_def_rating = np.random.normal(100, 10)
            team_recent_form = np.random.uniform(0, 1)  # Win rate in last 5 games
            opp_recent_form = np.random.uniform(0, 1)
            
            # Calculate winning probability based on features
            score_diff = (team_ppg - opp_ppg) * 1.5
            efficiency_diff = (team_off_rating - team_def_rating) - (opp_off_rating - opp_def_rating)
            shooting_diff = (team_fg_pct - opp_fg_pct) * 100
            win_pct_diff = (team_win_pct - opp_win_pct) * 50
            home_advantage = 3 if is_home == 1 else 0
            form_diff = (team_recent_form - opp_recent_form) * 10
            
            predicted_margin = (score_diff + efficiency_diff * 0.3 + shooting_diff * 0.5 + 
                              win_pct_diff + home_advantage + form_diff)
            
            # Convert to probability using sigmoid-like function
            win_prob = 1 / (1 + np.exp(-predicted_margin / 10))
            
            # Add some noise
            win_prob = np.clip(win_prob + np.random.normal(0, 0.1), 0, 1)
            
            features = [
                team_ppg, opp_ppg, team_fg_pct, opp_fg_pct,
                team_3p_pct, opp_3p_pct, team_ft_pct, opp_ft_pct,
                team_reb_pg, opp_reb_pg, team_ast_pg, opp_ast_pg,
                team_to_pg, opp_to_pg, team_stl_pg, opp_stl_pg,
                team_blk_pg, opp_blk_pg, team_win_pct, opp_win_pct,
                team_home_win_pct, opp_away_win_pct, is_home,
                team_pace, opp_pace, team_off_rating, opp_off_rating,
                team_def_rating, opp_def_rating, team_recent_form, opp_recent_form
            ]
            
            data.append(features + [win_prob])
        
        columns = self.feature_names + ['win_probability']
        return pd.DataFrame(data, columns=columns)
    
    def train(self, verbose=1):
        """
        Train the model on synthetic data
        """
        print("Generating training data...")
        data = self.generate_synthetic_data(n_samples=5000)
        
        X = data[self.feature_names].values
        y = data['win_probability'].values
        
        # Convert probabilities to binary outcomes for training
        y_binary = (y > 0.5).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining model on {len(X_train)} samples...")
        print(f"Testing on {len(X_test)} samples...")
        
        # Build and train neural network
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2 regularization
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
            verbose=verbose
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        test_acc = accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n✓ Model Training Complete!")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
    
    def predict(self, team_stats):
        """
        Predict win probability for a team
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        features = np.array([team_stats])
        features_scaled = self.scaler.transform(features)
        win_prob = self.model.predict_proba(features_scaled)[0][1]
        
        return win_prob
    
    def get_betting_insights(self, team_stats, spread, over_under=None, team_name="Team", opponent_name="Opponent"):
        """
        Generate betting insights based on predictions
        """
        win_prob = self.predict(team_stats)
        
        # Calculate implied probability from spread
        spread_win_prob = 0.5 + (spread / 50)  # Rough conversion
        spread_win_prob = np.clip(spread_win_prob, 0, 1)
        
        insights = {
            'win_probability': win_prob,
            'lose_probability': 1 - win_prob,
            'predicted_margin': (win_prob - 0.5) * 20,  # Rough margin estimate
            'spread': spread,
            'spread_value': win_prob - spread_win_prob,
            'over_under': over_under
        }
        
        print(f"\n{'='*60}")
        print(f"🏀 BETTING INSIGHTS: {team_name} vs {opponent_name}")
        print(f"{'='*60}")
        print(f"\n📊 Win Probability: {win_prob*100:.1f}%")
        print(f"📊 Loss Probability: {(1-win_prob)*100:.1f}%")
        print(f"📈 Predicted Margin: {insights['predicted_margin']:+.1f} points")
        
        print(f"\n💰 SPREAD ANALYSIS (Current Spread: {spread:+.1f})")
        if insights['spread_value'] > 0.05:
            print(f"   ✓ VALUE BET: {team_name} ({insights['spread_value']*100:.1f}% edge)")
            print(f"   Recommendation: Bet {team_name} {spread:+.1f}")
        elif insights['spread_value'] < -0.05:
            print(f"   ✓ VALUE BET: {opponent_name} ({-insights['spread_value']*100:.1f}% edge)")
            print(f"   Recommendation: Bet {opponent_name} {-spread:+.1f}")
        else:
            print(f"   ⚠ NO CLEAR VALUE (difference: {abs(insights['spread_value'])*100:.1f}%)")
            print(f"   Recommendation: Pass or bet small")
        
        if over_under:
            predicted_total = team_stats[0] + team_stats[1]  # Using PPG
            print(f"\n🎯 OVER/UNDER ANALYSIS (Line: {over_under})")
            print(f"   Predicted Total: {predicted_total:.1f} points")
            if predicted_total > over_under + 3:
                print(f"   ✓ Recommendation: OVER (by {predicted_total - over_under:.1f} points)")
            elif predicted_total < over_under - 3:
                print(f"   ✓ Recommendation: UNDER (by {over_under - predicted_total:.1f} points)")
            else:
                print(f"   ⚠ Recommendation: Pass (too close to call)")
        
        # Confidence rating
        confidence = abs(win_prob - 0.5) * 200
        print(f"\n⭐ Confidence Level: {confidence:.1f}%")
        if confidence > 70:
            print(f"   Very High Confidence")
        elif confidence > 50:
            print(f"   High Confidence")
        elif confidence > 30:
            print(f"   Moderate Confidence")
        else:
            print(f"   Low Confidence - Toss-up game")
        
        print(f"\n{'='*60}\n")
        
        return insights


def get_user_input():
    """
    Prompt user for game information
    """
    print("\n" + "="*60)
    print("🏀 NCAA BASKETBALL BETTING MODEL - GAME INPUT")
    print("="*60)
    
    team_name = input("\nEnter your team name: ").strip()
    opponent_name = input("Enter opponent team name: ").strip()
    
    print(f"\n--- {team_name} Statistics ---")
    team_ppg = float(input("Points per game: "))
    team_fg_pct = float(input("Field goal % (e.g., 45.5 for 45.5%): ")) / 100
    team_3p_pct = float(input("3-point % (e.g., 35.0 for 35%): ")) / 100
    team_ft_pct = float(input("Free throw % (e.g., 72.0 for 72%): ")) / 100
    team_reb_pg = float(input("Rebounds per game: "))
    team_ast_pg = float(input("Assists per game: "))
    team_to_pg = float(input("Turnovers per game: "))
    team_stl_pg = float(input("Steals per game: "))
    team_blk_pg = float(input("Blocks per game: "))
    team_win_pct = float(input("Win % (e.g., 75.0 for 75%): ")) / 100
    team_home_win_pct = float(input("Home win % (e.g., 85.0 for 85%): ")) / 100
    team_recent_form = float(input("Recent form - wins in last 5 (0-5): ")) / 5
    
    print(f"\n--- {opponent_name} Statistics ---")
    opp_ppg = float(input("Points per game: "))
    opp_fg_pct = float(input("Field goal % (e.g., 45.5 for 45.5%): ")) / 100
    opp_3p_pct = float(input("3-point % (e.g., 35.0 for 35%): ")) / 100
    opp_ft_pct = float(input("Free throw % (e.g., 72.0 for 72%): ")) / 100
    opp_reb_pg = float(input("Rebounds per game: "))
    opp_ast_pg = float(input("Assists per game: "))
    opp_to_pg = float(input("Turnovers per game: "))
    opp_stl_pg = float(input("Steals per game: "))
    opp_blk_pg = float(input("Blocks per game: "))
    opp_win_pct = float(input("Win % (e.g., 75.0 for 75%): ")) / 100
    opp_away_win_pct = float(input("Away win % (e.g., 65.0 for 65%): ")) / 100
    opp_recent_form = float(input("Recent form - wins in last 5 (0-5): ")) / 5
    
    print("\n--- Game Details ---")
    is_home = input(f"Is {team_name} playing at home? (yes/no): ").strip().lower()
    is_home = 1 if is_home in ['yes', 'y'] else 0
    
    # Advanced stats (can estimate if not available)
    print("\nAdvanced Stats (press Enter to use estimates):")
    try:
        team_pace = float(input(f"{team_name} pace (possessions/game) [default: 70]: ") or 70)
        opp_pace = float(input(f"{opponent_name} pace (possessions/game) [default: 70]: ") or 70)
        team_off_rating = float(input(f"{team_name} offensive rating [default: 110]: ") or 110)
        team_def_rating = float(input(f"{team_name} defensive rating [default: 100]: ") or 100)
        opp_off_rating = float(input(f"{opponent_name} offensive rating [default: 110]: ") or 110)
        opp_def_rating = float(input(f"{opponent_name} defensive rating [default: 100]: ") or 100)
    except:
        team_pace = opp_pace = 70
        team_off_rating = opp_off_rating = 110
        team_def_rating = opp_def_rating = 100
    
    print("\n--- Betting Lines ---")
    spread = float(input(f"Point spread for {team_name} (e.g., -5.5 or +3.5): "))
    over_under_input = input("Over/Under total (press Enter to skip): ").strip()
    over_under = float(over_under_input) if over_under_input else None
    
    # Compile features
    team_stats = [
        team_ppg, opp_ppg, team_fg_pct, opp_fg_pct,
        team_3p_pct, opp_3p_pct, team_ft_pct, opp_ft_pct,
        team_reb_pg, opp_reb_pg, team_ast_pg, opp_ast_pg,
        team_to_pg, opp_to_pg, team_stl_pg, opp_stl_pg,
        team_blk_pg, opp_blk_pg, team_win_pct, opp_win_pct,
        team_home_win_pct, opp_away_win_pct, is_home,
        team_pace, opp_pace, team_off_rating, opp_off_rating,
        team_def_rating, opp_def_rating, team_recent_form, opp_recent_form
    ]
    
    return team_stats, spread, over_under, team_name, opponent_name


def main():
    """
    Main function to run the betting model
    """
    print("\n🏀 NCAA MEN'S BASKETBALL DEEP LEARNING BETTING MODEL")
    print("=" * 60)
    
    # Initialize and train model
    model = NCAABettingModel()
    print("\nInitializing and training model...")
    model.train(verbose=0)
    
    # Get user input
    team_stats, spread, over_under, team_name, opponent_name = get_user_input()
    
    # Generate predictions and insights
    insights = model.get_betting_insights(
        team_stats, spread, over_under, team_name, opponent_name
    )
    
    # Ask if user wants to analyze another game
    while True:
        another = input("Analyze another game? (yes/no): ").strip().lower()
        if another in ['yes', 'y']:
            team_stats, spread, over_under, team_name, opponent_name = get_user_input()
            insights = model.get_betting_insights(
                team_stats, spread, over_under, team_name, opponent_name
            )
        else:
            print("\n✓ Thank you for using the NCAA Basketball Betting Model!")
            print("Remember: Bet responsibly! This model is for educational purposes.\n")
            break


if __name__ == "__main__":
    main()
