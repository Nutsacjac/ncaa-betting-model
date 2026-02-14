import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For real NBA data, you'll need to install:
# pip install nba_api requests

class NBADataLoader:
    """
    Loads real NBA data from various sources.
    """
    
    def __init__(self):
        self.team_stats = None
        self.games = None
    
    def load_from_nba_api(self, season='2024-25'):
        """
        Load data using the official NBA API (via nba_api package).
        Install: pip install nba_api
        """
        try:
            from nba_api.stats.endpoints import leaguegamefinder, teamdashboardbygeneralsplits
            from nba_api.stats.static import teams
            
            print(f"Loading NBA data for {season} season...")
            
            # Get all NBA teams
            nba_teams = teams.get_teams()
            team_dict = {team['id']: team['abbreviation'] for team in nba_teams}
            
            # Get team stats
            team_stats_list = []
            for team in nba_teams[:5]:  # Limit to 5 teams for demo (remove [:5] for all)
                team_id = team['id']
                team_name = team['abbreviation']
                
                print(f"Fetching stats for {team_name}...")
                
                dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                    team_id=team_id,
                    season=season
                )
                
                stats = dashboard.get_data_frames()[0]
                if not stats.empty:
                    team_stats_list.append({
                        'team_id': team_id,
                        'team_name': team_name,
                        'win_pct': stats['W_PCT'].iloc[0],
                        'pts_per_game': stats['PTS'].iloc[0],
                        'fg_pct': stats['FG_PCT'].iloc[0],
                        'three_pct': stats['FG3_PCT'].iloc[0],
                        'def_rating': stats['DEF_RATING'].iloc[0] if 'DEF_RATING' in stats else 110.0,
                    })
            
            self.team_stats = pd.DataFrame(team_stats_list)
            
            # Get games
            print("\nFetching game data...")
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable='00'
            )
            games = gamefinder.get_data_frames()[0]
            
            self.games = games
            print(f"Loaded {len(games)} game records")
            
            return self.team_stats, self.games
            
        except ImportError:
            print("ERROR: nba_api not installed. Install with: pip install nba_api")
            return None, None
        except Exception as e:
            print(f"Error loading NBA API data: {e}")
            return None, None
    
    def load_from_api_url(self, api_url="https://www.balldontlie.io/api/v1"):
        """
        Load data from Ball Don't Lie API (free, no key required).
        Documentation: https://www.balldontlie.io/
        """
        try:
            import requests
            
            print("Loading data from Ball Don't Lie API...")
            
            # Get teams
            response = requests.get(f"{api_url}/teams")
            teams = response.json()['data']
            
            # Get current season stats
            team_stats_list = []
            for team in teams[:5]:  # Limit for demo
                team_id = team['id']
                team_name = team['abbreviation']
                
                # Get season averages (you'd need to calculate from games)
                print(f"Fetching stats for {team_name}...")
                
                # This is simplified - you'd calculate these from actual games
                team_stats_list.append({
                    'team_id': team_id,
                    'team_name': team_name,
                    'full_name': team['full_name']
                })
            
            self.team_stats = pd.DataFrame(team_stats_list)
            print(f"Loaded {len(self.team_stats)} teams")
            
            return self.team_stats
            
        except Exception as e:
            print(f"Error loading from API: {e}")
            return None
    
    def load_from_csv(self, stats_file, games_file=None):
        """
        Load data from CSV files.
        You can download NBA data from:
        - Basketball Reference: https://www.basketball-reference.com/
        - Kaggle: https://www.kaggle.com/datasets (search for NBA)
        """
        try:
            print(f"Loading team stats from {stats_file}...")
            self.team_stats = pd.read_csv(stats_file)
            
            if games_file:
                print(f"Loading games from {games_file}...")
                self.games = pd.read_csv(games_file)
            
            print("Data loaded successfully!")
            return self.team_stats, self.games
            
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            return None, None
    
    def prepare_training_data(self):
        """
        Convert raw game data into training features.
        """
        if self.games is None or self.team_stats is None:
            print("No data loaded. Load data first.")
            return None
        
        # This is a simplified version - you'd expand this based on your data structure
        training_data = []
        
        # Group games by matchup
        for idx, game in self.games.iterrows():
            # Extract team IDs and stats
            # Structure depends on your data source
            pass
        
        return pd.DataFrame(training_data)


class NBAWinnerPredictor:
    """
    NBA game winner prediction model using team statistics.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def create_sample_data(self, n_games=1000):
        """
        Creates sample training data with team statistics.
        In production, you'd load real NBA data from an API or database.
        """
        np.random.seed(42)
        
        data = {
            # Home team stats
            'home_win_pct': np.random.uniform(0.3, 0.7, n_games),
            'home_pts_per_game': np.random.uniform(100, 120, n_games),
            'home_fg_pct': np.random.uniform(0.42, 0.50, n_games),
            'home_three_pct': np.random.uniform(0.32, 0.40, n_games),
            'home_def_rating': np.random.uniform(105, 115, n_games),
            'home_rest_days': np.random.randint(0, 5, n_games),
            
            # Away team stats
            'away_win_pct': np.random.uniform(0.3, 0.7, n_games),
            'away_pts_per_game': np.random.uniform(100, 120, n_games),
            'away_fg_pct': np.random.uniform(0.42, 0.50, n_games),
            'away_three_pct': np.random.uniform(0.32, 0.40, n_games),
            'away_def_rating': np.random.uniform(105, 115, n_games),
            'away_rest_days': np.random.randint(0, 5, n_games),
        }
        
        df = pd.DataFrame(data)
        
        # Create outcome: home team wins (1) or loses (0)
        # Weight by team statistics to create realistic outcomes
        win_prob = (
            0.35 * (df['home_win_pct'] - df['away_win_pct']) +
            0.15 * (df['home_pts_per_game'] - df['away_pts_per_game']) / 20 +
            0.15 * (df['home_fg_pct'] - df['away_fg_pct']) * 10 +
            0.10 * (df['away_def_rating'] - df['home_def_rating']) / 10 +
            0.15  # Home court advantage
        )
        
        df['home_win'] = (win_prob + np.random.normal(0, 0.15, n_games) > 0.5).astype(int)
        
        return df
    
    
    def train_with_real_data(self, data_loader):
        """
        Train model using real NBA data from a data loader.
        """
        training_data = data_loader.prepare_training_data()
        
        if training_data is not None and not training_data.empty:
            return self.train(training_data)
        else:
            print("No training data available. Using sample data instead.")
            sample_data = self.create_sample_data()
            return self.train(sample_data)
    
    def train(self, df):
        """
        Train the model on historical game data.
        """
        X = df.drop('home_win', axis=1)
        y = df['home_win']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("Model Training Complete!")
        print(f"Test Accuracy: {accuracy:.2%}")
        print("\nFeature Importance:")
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.to_string(index=False))
        
        return accuracy
    
    def predict_game(self, home_stats, away_stats):
        """
        Predict the winner of a single game.
        
        Args:
            home_stats: dict with keys matching training features
            away_stats: dict with keys matching training features
        
        Returns:
            Prediction and probability
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        game_data = {
            'home_win_pct': home_stats['win_pct'],
            'home_pts_per_game': home_stats['pts_per_game'],
            'home_fg_pct': home_stats['fg_pct'],
            'home_three_pct': home_stats['three_pct'],
            'home_def_rating': home_stats['def_rating'],
            'home_rest_days': home_stats['rest_days'],
            'away_win_pct': away_stats['win_pct'],
            'away_pts_per_game': away_stats['pts_per_game'],
            'away_fg_pct': away_stats['fg_pct'],
            'away_three_pct': away_stats['three_pct'],
            'away_def_rating': away_stats['def_rating'],
            'away_rest_days': away_stats['rest_days'],
        }
        
        X = pd.DataFrame([game_data])
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        winner = "Home Team" if prediction == 1 else "Away Team"
        confidence = max(probability) * 100
        
        return {
            'winner': winner,
            'confidence': confidence,
            'home_win_prob': probability[1] * 100,
            'away_win_prob': probability[0] * 100
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("NBA GAME WINNER PREDICTION MODEL - REAL DATA INTEGRATION")
    print("=" * 60)
    print()
    
    # Initialize predictor and data loader
    predictor = NBAWinnerPredictor()
    data_loader = NBADataLoader()
    
    print("OPTION 1: Using NBA API (Official)")
    print("-" * 60)
    print("Uncomment the code below to use real NBA API data:")
    print("team_stats, games = data_loader.load_from_nba_api(season='2024-25')")
    print("predictor.train_with_real_data(data_loader)")
    print()
    
    print("OPTION 2: Using Ball Don't Lie API (Free)")
    print("-" * 60)
    print("Uncomment the code below to use Ball Don't Lie API:")
    print("# team_stats = data_loader.load_from_api_url()")
    print()
    
    print("OPTION 3: Using CSV Files")
    print("-" * 60)
    print("Download NBA data from Basketball Reference or Kaggle, then:")
    print("# team_stats, games = data_loader.load_from_csv('team_stats.csv', 'games.csv')")
    print()
    
    print("=" * 60)
    print("DEMO: Using Sample Data")
    print("=" * 60)
    print()
    
    # For now, use sample data
    print("Training model on sample data...")
    print()
    sample_data = predictor.create_sample_data(n_games=1000)
    predictor.train(sample_data)
    
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    print()
    
    # Example game prediction
    home_team = {
        'win_pct': 0.650,
        'pts_per_game': 115.2,
        'fg_pct': 0.475,
        'three_pct': 0.380,
        'def_rating': 108.5,
        'rest_days': 2
    }
    
    away_team = {
        'win_pct': 0.520,
        'pts_per_game': 110.8,
        'fg_pct': 0.455,
        'three_pct': 0.360,
        'def_rating': 111.2,
        'rest_days': 1
    }
    
    result = predictor.predict_game(home_team, away_team)
    
    print(f"Predicted Winner: {result['winner']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"\nHome Team Win Probability: {result['home_win_prob']:.1f}%")
    print(f"Away Team Win Probability: {result['away_win_prob']:.1f}%")
    
    print("\n" + "=" * 60)
    print("SETUP INSTRUCTIONS FOR REAL DATA")
    print("=" * 60)
    print("""
1. NBA API (Official - Most Comprehensive):
   pip install nba_api
   
   Provides: Team stats, player stats, game results, advanced metrics
   Free, but rate-limited
   
2. Ball Don't Lie API (Simple - No API Key):
   pip install requests
   
   Endpoint: https://www.balldontlie.io/api/v1
   Provides: Basic stats, games, teams
   Free tier available
   
3. SportsData.io (Commercial - Very Detailed):
   Requires API key (paid)
   
   Provides: Real-time scores, detailed stats, injuries, odds
   Visit: https://sportsdata.io/
   
4. CSV Files (Offline):
   Download from:
   - Basketball Reference: https://www.basketball-reference.com/
   - Kaggle: https://www.kaggle.com/datasets
   - NBA Stats: https://stats.nba.com/
   
   Export as CSV and load with data_loader.load_from_csv()
""")
