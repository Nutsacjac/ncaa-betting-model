import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

API_KEY = "0a8137b-042b-4368-9d15-1a6baa47d748"
BASE_URL = "https://api.balldontlie.io"




# Install required package: pip install requests

class BallDontLieDataLoader:
    """
    Loads real NBA data from Ball Don't Lie API (Free, no key required).
    API Documentation: https://docs.balldontlie.io/
    """
    
    def __init__(self):
        self.base_url = "https://api.balldontlie.io/"
        self.teams = None
        self.games = None
        self.team_stats = None
    
    def get_all_teams(self):
        """Fetch all NBA teams."""
        try:
            print("Fetching NBA teams...")
            response = requests.get(f"{self.base_url}/teams")
            
            if response.status_code == 200:
                data = response.json()
                self.teams = pd.DataFrame(data['data'])
                print(f"✓ Loaded {len(self.teams)} teams")
                return self.teams
            else:
                print(f"Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching teams: {e}")
            return None
    
    def get_games(self, season=2025, start_date=None, end_date=None):
        """
        Fetch games for a specific season.
        Season format: 2024 for 2024-25 season
        """
        try:
            print(f"Fetching games for {season}-{season+1} season...")
            
            all_games = []
            page = 1
            per_page = 100
            
            while True:
                params = {
                    'seasons[]': season,
                    'per_page': per_page,
                    'page': page
                }
                
                if start_date:
                    params['start_date'] = start_date
                if end_date:
                    params['end_date'] = end_date
                
                response = requests.get(f"{self.base_url}/games", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    games = data['data']
                    
                    if not games:
                        break
                    
                    all_games.extend(games)
                    print(f"  Fetched page {page} ({len(all_games)} games so far)...")
                    
                    # Check if there are more pages
                    if len(games) < per_page:
                        break
                    
                    page += 1
                    time.sleep(0.6)  # Rate limiting - be nice to the API
                else:
                    print(f"Error: {response.status_code}")
                    break
            
            self.games = pd.DataFrame(all_games)
            print(f"✓ Loaded {len(self.games)} games total")
            return self.games
            
        except Exception as e:
            print(f"Error fetching games: {e}")
            return None
    
    def calculate_team_stats(self):
        """Calculate team statistics from game data."""
        if self.games is None or self.games.empty:
            print("No games loaded. Fetch games first.")
            return None
        
        print("\nCalculating team statistics from game data...")
        
        team_stats_dict = defaultdict(lambda: {
            'games': 0,
            'wins': 0,
            'total_pts': 0,
            'total_pts_allowed': 0,
            'home_games': 0,
            'home_wins': 0
        })
        
        # Process each game
        for _, game in self.games.iterrows():
            # Skip games that haven't been played
            if game['status'] != 'Final':
                continue
            
            home_team_id = game['home_team']['id']
            away_team_id = game['visitor_team']['id']
            home_score = game['home_team_score']
            away_score = game['visitor_team_score']
            
            # Home team stats
            team_stats_dict[home_team_id]['games'] += 1
            team_stats_dict[home_team_id]['home_games'] += 1
            team_stats_dict[home_team_id]['total_pts'] += home_score
            team_stats_dict[home_team_id]['total_pts_allowed'] += away_score
            if home_score > away_score:
                team_stats_dict[home_team_id]['wins'] += 1
                team_stats_dict[home_team_id]['home_wins'] += 1
            
            # Away team stats
            team_stats_dict[away_team_id]['games'] += 1
            team_stats_dict[away_team_id]['total_pts'] += away_score
            team_stats_dict[away_team_id]['total_pts_allowed'] += home_score
            if away_score > home_score:
                team_stats_dict[away_team_id]['wins'] += 1
        
        # Convert to DataFrame with calculated metrics
        stats_list = []
        for team_id, stats in team_stats_dict.items():
            if stats['games'] > 0:
                team_info = self.teams[self.teams['id'] == team_id].iloc[0]
                
                stats_list.append({
                    'team_id': team_id,
                    'team_name': team_info['abbreviation'],
                    'full_name': team_info['full_name'],
                    'games_played': stats['games'],
                    'wins': stats['wins'],
                    'losses': stats['games'] - stats['wins'],
                    'win_pct': stats['wins'] / stats['games'],
                    'pts_per_game': stats['total_pts'] / stats['games'],
                    'pts_allowed_per_game': stats['total_pts_allowed'] / stats['games'],
                    'point_diff': (stats['total_pts'] - stats['total_pts_allowed']) / stats['games'],
                    'home_win_pct': stats['home_wins'] / stats['home_games'] if stats['home_games'] > 0 else 0
                })
        
        self.team_stats = pd.DataFrame(stats_list).sort_values('win_pct', ascending=False)
        print(f"✓ Calculated stats for {len(self.team_stats)} teams")
        
        return self.team_stats
    
    def prepare_training_data(self):
        """Convert game data into ML training features."""
        if self.games is None or self.team_stats is None:
            print("Load games and calculate team stats first.")
            return None
        
        print("\nPreparing training data...")
        
        training_data = []
        
        # Create a dictionary for quick team stats lookup
        team_stats_dict = {
            row['team_id']: row 
            for _, row in self.team_stats.iterrows()
        }
        
        # Process each completed game
        for _, game in self.games.iterrows():
            if game['status'] != 'Final':
                continue
            
            home_team_id = game['home_team']['id']
            away_team_id = game['visitor_team']['id']
            
            if home_team_id not in team_stats_dict or away_team_id not in team_stats_dict:
                continue
            
            home_stats = team_stats_dict[home_team_id]
            away_stats = team_stats_dict[away_team_id]
            
            home_score = game['home_team_score']
            away_score = game['visitor_team_score']
            home_win = 1 if home_score > away_score else 0
            
            training_data.append({
                'home_win_pct': home_stats['win_pct'],
                'home_pts_per_game': home_stats['pts_per_game'],
                'home_pts_allowed': home_stats['pts_allowed_per_game'],
                'home_point_diff': home_stats['point_diff'],
                'away_win_pct': away_stats['win_pct'],
                'away_pts_per_game': away_stats['pts_per_game'],
                'away_pts_allowed': away_stats['pts_allowed_per_game'],
                'away_point_diff': away_stats['point_diff'],
                'home_win': home_win
            })
        
        df = pd.DataFrame(training_data)
        print(f"✓ Prepared {len(df)} training samples")
        
        return df


class NBAWinnerPredictor:
    """NBA game winner prediction model."""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        self.is_trained = False
        self.feature_names = None
        
    def train(self, df):
        """Train the model on game data."""
        X = df.drop('home_win', axis=1)
        y = df['home_win']
        
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("=" * 60)
        print("MODEL TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Test Accuracy: {accuracy:.2%}")
        print("\nFeature Importance:")
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']:<25} {row['importance']:.4f}")
        
        return accuracy
    
    def predict_game(self, home_team_name, away_team_name, team_stats):
        """
        Predict winner given team names.
        
        Args:
            home_team_name: Home team abbreviation (e.g., 'LAL')
            away_team_name: Away team abbreviation (e.g., 'BOS')
            team_stats: DataFrame with team statistics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Find team stats
        home_stats = team_stats[team_stats['team_name'] == home_team_name]
        away_stats = team_stats[team_stats['team_name'] == away_team_name]
        
        if home_stats.empty or away_stats.empty:
            raise ValueError(f"Could not find stats for {home_team_name} or {away_team_name}")
        
        home_stats = home_stats.iloc[0]
        away_stats = away_stats.iloc[0]
        
        # Prepare features
        game_data = {
            'home_win_pct': home_stats['win_pct'],
            'home_pts_per_game': home_stats['pts_per_game'],
            'home_pts_allowed': home_stats['pts_allowed_per_game'],
            'home_point_diff': home_stats['point_diff'],
            'away_win_pct': away_stats['win_pct'],
            'away_pts_per_game': away_stats['pts_per_game'],
            'away_pts_allowed': away_stats['pts_allowed_per_game'],
            'away_point_diff': away_stats['point_diff'],
        }
        
        X = pd.DataFrame([game_data])
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        winner = f"{home_stats['full_name']}" if prediction == 1 else f"{away_stats['full_name']}"
        confidence = max(probability) * 100
        
        return {
            'home_team': home_stats['full_name'],
            'away_team': away_stats['full_name'],
            'winner': winner,
            'confidence': confidence,
            'home_win_prob': probability[1] * 100,
            'away_win_prob': probability[0] * 100,
            'home_record': f"{home_stats['wins']}-{home_stats['losses']}",
            'away_record': f"{away_stats['wins']}-{away_stats['losses']}"
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("NBA WINNER PREDICTOR - BALL DON'T LIE API")
    print("=" * 60)
    print()
    
    # Initialize
    data_loader = BallDontLieDataLoader()
    predictor = NBAWinnerPredictor()
    
    # Step 1: Load teams
    teams = data_loader.get_all_teams()
    
    if teams is not None:
        print("\nSample teams:")
        print(teams[['id', 'abbreviation', 'full_name', 'city']].head(10))
    
    # Step 2: Load games (last 30 days for faster demo)
    # For full season, use: data_loader.get_games(season=2024)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"\nFetching games from {start_date} to {end_date}...")
    games = data_loader.get_games(season=2024, start_date=start_date, end_date=end_date)
    
    if games is not None and not games.empty:
        # Step 3: Calculate team statistics
        team_stats = data_loader.calculate_team_stats()
        
        if team_stats is not None:
            print("\n" + "=" * 60)
            print("TOP 10 TEAMS BY WIN PERCENTAGE")
            print("=" * 60)
            print(team_stats[['team_name', 'full_name', 'wins', 'losses', 'win_pct', 'pts_per_game']].head(10).to_string(index=False))
            
            # Step 4: Prepare training data
            training_data = data_loader.prepare_training_data()
            
            if training_data is not None and len(training_data) > 50:
                # Step 5: Train model
                predictor.train(training_data)
                
                # Step 6: Make predictions
                print("\n" + "=" * 60)
                print("EXAMPLE PREDICTIONS")
                print("=" * 60)
                
                # Get top teams for demo
                if len(team_stats) >= 2:
                    top_teams = team_stats.head(5)['team_name'].tolist()
                    
                    if len(top_teams) >= 2:
                        home = top_teams[0]
                        away = top_teams[1]
                        
                        print(f"\n📊 Prediction: {home} (Home) vs {away} (Away)")
                        print("-" * 60)
                        
                        result = predictor.predict_game(home, away, team_stats)
                        
                        print(f"Home: {result['home_team']} ({result['home_record']})")
                        print(f"Away: {result['away_team']} ({result['away_record']})")
                        print(f"\n🏆 Predicted Winner: {result['winner']}")
                        print(f"Confidence: {result['confidence']:.1f}%")
                        print(f"\nWin Probabilities:")
                        print(f"  {result['home_team']}: {result['home_win_prob']:.1f}%")
                        print(f"  {result['away_team']}: {result['away_win_prob']:.1f}%")
            else:
                print("\n⚠️  Not enough games to train. Try a longer date range.")
    else:
        print("\n⚠️  No games loaded. Check your internet connection or try again later.")
    
    print("\n" + "=" * 60)
    print("USAGE TIPS")
    print("=" * 60)
    print("""
To predict any matchup:
  result = predictor.predict_game('LAL', 'BOS', team_stats)

To get full season data:
  games = data_loader.get_games(season=2024)

Available team abbreviations: ATL, BOS, BKN, CHA, CHI, CLE, DAL, DEN, DET,
GSW, HOU, IND, LAC, LAL, MEM, MIA, MIL, MIN, NOP, NYK, OKC, ORL, PHI, PHX,
POR, SAC, SAS, TOR, UTA, WAS
""")