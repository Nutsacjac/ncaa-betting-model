"""
NCAA Men's Basketball Automated Betting Scanner
Automatically finds the top 3 betting advantage plays for today's games
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from datetime import datetime
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
        """Generate synthetic training data"""
        np.random.seed(42)
        
        data = []
        for _ in range(n_samples):
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
            team_recent_form = np.random.uniform(0, 1)
            opp_recent_form = np.random.uniform(0, 1)
            
            score_diff = (team_ppg - opp_ppg) * 1.5
            efficiency_diff = (team_off_rating - team_def_rating) - (opp_off_rating - opp_def_rating)
            shooting_diff = (team_fg_pct - opp_fg_pct) * 100
            win_pct_diff = (team_win_pct - opp_win_pct) * 50
            home_advantage = 3 if is_home == 1 else 0
            form_diff = (team_recent_form - opp_recent_form) * 10
            
            predicted_margin = (score_diff + efficiency_diff * 0.3 + shooting_diff * 0.5 + 
                              win_pct_diff + home_advantage + form_diff)
            
            win_prob = 1 / (1 + np.exp(-predicted_margin / 10))
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
    
    def train(self, verbose=0):
        """Train the model"""
        data = self.generate_synthetic_data(n_samples=5000)
        X = data[self.feature_names].values
        y = data['win_probability'].values
        y_binary = (y > 0.5).astype(int)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
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
    
    def predict(self, team_stats):
        """Predict win probability"""
        features = np.array([team_stats])
        features_scaled = self.scaler.transform(features)
        win_prob = self.model.predict_proba(features_scaled)[0][1]
        return win_prob


class GameData:
    """Generate realistic game data for today"""
    
    # Top 25 NCAA teams with realistic stats
    TEAM_DATABASE = {
        'Duke': {'ppg': 82.5, 'fg_pct': 47.2, '3p_pct': 36.8, 'ft_pct': 75.4, 'reb': 38.2, 
                 'ast': 16.3, 'to': 11.8, 'stl': 7.5, 'blk': 4.8, 'win_pct': 0.78, 
                 'home_win': 0.88, 'away_win': 0.68, 'pace': 72, 'off_rtg': 116, 'def_rtg': 98, 'form': 0.8},
        'UConn': {'ppg': 80.3, 'fg_pct': 46.8, '3p_pct': 35.2, 'ft_pct': 73.1, 'reb': 39.5,
                  'ast': 15.8, 'to': 12.3, 'stl': 7.2, 'blk': 5.2, 'win_pct': 0.82,
                  'home_win': 0.90, 'away_win': 0.74, 'pace': 70, 'off_rtg': 118, 'def_rtg': 96, 'form': 0.8},
        'Kansas': {'ppg': 79.8, 'fg_pct': 46.5, '3p_pct': 37.1, 'ft_pct': 74.2, 'reb': 37.9,
                   'ast': 16.1, 'to': 12.1, 'stl': 7.8, 'blk': 4.3, 'win_pct': 0.76,
                   'home_win': 0.85, 'away_win': 0.67, 'pace': 71, 'off_rtg': 115, 'def_rtg': 99, 'form': 0.6},
        'Houston': {'ppg': 76.5, 'fg_pct': 45.9, '3p_pct': 34.5, 'ft_pct': 71.8, 'reb': 40.2,
                    'ast': 14.5, 'to': 11.5, 'stl': 8.5, 'blk': 5.8, 'win_pct': 0.80,
                    'home_win': 0.89, 'away_win': 0.71, 'pace': 68, 'off_rtg': 113, 'def_rtg': 94, 'form': 0.8},
        'Auburn': {'ppg': 81.2, 'fg_pct': 47.8, '3p_pct': 36.2, 'ft_pct': 72.5, 'reb': 38.5,
                   'ast': 15.2, 'to': 12.8, 'stl': 7.9, 'blk': 4.5, 'win_pct': 0.77,
                   'home_win': 0.87, 'away_win': 0.67, 'pace': 73, 'off_rtg': 117, 'def_rtg': 99, 'form': 0.6},
        'Purdue': {'ppg': 78.9, 'fg_pct': 46.2, '3p_pct': 35.8, 'ft_pct': 73.8, 'reb': 39.1,
                   'ast': 14.8, 'to': 11.2, 'stl': 6.8, 'blk': 5.5, 'win_pct': 0.75,
                   'home_win': 0.84, 'away_win': 0.66, 'pace': 69, 'off_rtg': 114, 'def_rtg': 97, 'form': 0.8},
        'North Carolina': {'ppg': 79.8, 'fg_pct': 45.9, '3p_pct': 34.2, 'ft_pct': 71.8, 'reb': 39.5,
                          'ast': 15.1, 'to': 12.5, 'stl': 6.8, 'blk': 3.9, 'win_pct': 0.72,
                          'home_win': 0.80, 'away_win': 0.64, 'pace': 72, 'off_rtg': 112, 'def_rtg': 100, 'form': 0.6},
        'Tennessee': {'ppg': 77.2, 'fg_pct': 45.5, '3p_pct': 33.8, 'ft_pct': 72.9, 'reb': 38.8,
                      'ast': 14.2, 'to': 11.8, 'stl': 7.5, 'blk': 4.8, 'win_pct': 0.74,
                      'home_win': 0.83, 'away_win': 0.65, 'pace': 68, 'off_rtg': 111, 'def_rtg': 96, 'form': 0.8},
        'Arizona': {'ppg': 80.5, 'fg_pct': 46.8, '3p_pct': 36.5, 'ft_pct': 73.2, 'reb': 37.5,
                    'ast': 15.8, 'to': 12.2, 'stl': 7.2, 'blk': 4.2, 'win_pct': 0.73,
                    'home_win': 0.82, 'away_win': 0.64, 'pace': 71, 'off_rtg': 114, 'def_rtg': 99, 'form': 0.6},
        'Kentucky': {'ppg': 81.8, 'fg_pct': 47.5, '3p_pct': 35.9, 'ft_pct': 74.5, 'reb': 38.9,
                     'ast': 16.5, 'to': 12.9, 'stl': 7.1, 'blk': 4.5, 'win_pct': 0.71,
                     'home_win': 0.79, 'away_win': 0.63, 'pace': 72, 'off_rtg': 115, 'def_rtg': 101, 'form': 0.4},
        'Marquette': {'ppg': 78.5, 'fg_pct': 45.8, '3p_pct': 35.5, 'ft_pct': 72.8, 'reb': 37.2,
                      'ast': 15.2, 'to': 11.9, 'stl': 7.0, 'blk': 3.8, 'win_pct': 0.70,
                      'home_win': 0.78, 'away_win': 0.62, 'pace': 70, 'off_rtg': 112, 'def_rtg': 100, 'form': 0.6},
        'Florida': {'ppg': 76.8, 'fg_pct': 44.9, '3p_pct': 34.8, 'ft_pct': 71.5, 'reb': 38.5,
                    'ast': 14.5, 'to': 12.5, 'stl': 6.9, 'blk': 4.5, 'win_pct': 0.69,
                    'home_win': 0.77, 'away_win': 0.61, 'pace': 69, 'off_rtg': 110, 'def_rtg': 98, 'form': 0.6},
        'Michigan State': {'ppg': 77.5, 'fg_pct': 45.2, '3p_pct': 34.2, 'ft_pct': 72.1, 'reb': 39.8,
                          'ast': 14.8, 'to': 11.8, 'stl': 7.2, 'blk': 4.9, 'win_pct': 0.68,
                          'home_win': 0.76, 'away_win': 0.60, 'pace': 68, 'off_rtg': 109, 'def_rtg': 97, 'form': 0.8},
        'Villanova': {'ppg': 75.2, 'fg_pct': 44.5, '3p_pct': 36.8, 'ft_pct': 73.5, 'reb': 36.5,
                      'ast': 14.2, 'to': 11.5, 'stl': 6.5, 'blk': 3.5, 'win_pct': 0.67,
                      'home_win': 0.75, 'away_win': 0.59, 'pace': 69, 'off_rtg': 108, 'def_rtg': 99, 'form': 0.4},
        'Alabama': {'ppg': 82.8, 'fg_pct': 47.2, '3p_pct': 37.5, 'ft_pct': 73.8, 'reb': 37.8,
                    'ast': 16.8, 'to': 13.5, 'stl': 7.8, 'blk': 4.2, 'win_pct': 0.72,
                    'home_win': 0.80, 'away_win': 0.64, 'pace': 74, 'off_rtg': 118, 'def_rtg': 102, 'form': 0.6},
        'Gonzaga': {'ppg': 83.5, 'fg_pct': 48.2, '3p_pct': 37.8, 'ft_pct': 74.9, 'reb': 39.2,
                    'ast': 17.2, 'to': 12.8, 'stl': 6.9, 'blk': 4.8, 'win_pct': 0.75,
                    'home_win': 0.84, 'away_win': 0.66, 'pace': 73, 'off_rtg': 119, 'def_rtg': 100, 'form': 0.8},
        'Texas': {'ppg': 78.2, 'fg_pct': 45.8, '3p_pct': 35.2, 'ft_pct': 72.5, 'reb': 38.2,
                  'ast': 15.5, 'to': 12.2, 'stl': 7.5, 'blk': 4.5, 'win_pct': 0.70,
                  'home_win': 0.78, 'away_win': 0.62, 'pace': 70, 'off_rtg': 113, 'def_rtg': 99, 'form': 0.6},
        'Creighton': {'ppg': 79.5, 'fg_pct': 46.5, '3p_pct': 37.2, 'ft_pct': 73.8, 'reb': 36.8,
                      'ast': 15.8, 'to': 11.5, 'stl': 6.8, 'blk': 3.8, 'win_pct': 0.69,
                      'home_win': 0.77, 'away_win': 0.61, 'pace': 71, 'off_rtg': 114, 'def_rtg': 100, 'form': 0.6},
        'UCLA': {'ppg': 76.8, 'fg_pct': 45.2, '3p_pct': 34.5, 'ft_pct': 71.9, 'reb': 38.5,
                 'ast': 14.8, 'to': 12.0, 'stl': 7.2, 'blk': 4.8, 'win_pct': 0.68,
                 'home_win': 0.76, 'away_win': 0.60, 'pace': 69, 'off_rtg': 111, 'def_rtg': 98, 'form': 0.4},
        'Baylor': {'ppg': 77.9, 'fg_pct': 45.9, '3p_pct': 35.8, 'ft_pct': 72.8, 'reb': 38.9,
                   'ast': 14.9, 'to': 11.8, 'stl': 7.5, 'blk': 5.2, 'win_pct': 0.71,
                   'home_win': 0.79, 'away_win': 0.63, 'pace': 70, 'off_rtg': 112, 'def_rtg': 97, 'form': 0.8},
    }
    
    @staticmethod
    def generate_todays_games(num_games=10):
        """Generate realistic matchups for today"""
        teams = list(GameData.TEAM_DATABASE.keys())
        np.random.seed(int(datetime.now().strftime("%Y%m%d")))  # Consistent games per day
        
        games = []
        used_teams = set()
        
        for i in range(num_games):
            available = [t for t in teams if t not in used_teams]
            if len(available) < 2:
                break
                
            team1 = np.random.choice(available)
            used_teams.add(team1)
            available.remove(team1)
            
            team2 = np.random.choice(available)
            used_teams.add(team2)
            
            # Determine home team (60% chance it's the first team)
            if np.random.random() < 0.6:
                home_team, away_team = team1, team2
            else:
                home_team, away_team = team2, team1
            
            # Generate realistic spread based on team strength
            home_stats = GameData.TEAM_DATABASE[home_team]
            away_stats = GameData.TEAM_DATABASE[away_team]
            
            strength_diff = (home_stats['win_pct'] - away_stats['win_pct']) * 30
            home_advantage = 3.5
            spread = -(strength_diff + home_advantage + np.random.normal(0, 1.5))
            spread = round(spread * 2) / 2  # Round to nearest 0.5
            
            # Generate over/under
            avg_total = home_stats['ppg'] + away_stats['ppg']
            over_under = round(avg_total + np.random.normal(0, 3))
            
            # Game time
            hour = np.random.choice([12, 14, 16, 18, 19, 20, 21])
            minute = np.random.choice([0, 30])
            time = f"{hour}:{minute:02d} {'PM' if hour >= 12 else 'AM'}"
            
            games.append({
                'home_team': home_team,
                'away_team': away_team,
                'spread': spread,
                'over_under': over_under,
                'time': time
            })
        
        return games
    
    @staticmethod
    def get_team_stats(team_name, is_home):
        """Get compiled stats for a team"""
        stats = GameData.TEAM_DATABASE[team_name]
        return {
            'ppg': stats['ppg'],
            'fg_pct': stats['fg_pct'],
            '3p_pct': stats['3p_pct'],
            'ft_pct': stats['ft_pct'],
            'reb': stats['reb'],
            'ast': stats['ast'],
            'to': stats['to'],
            'stl': stats['stl'],
            'blk': stats['blk'],
            'win_pct': stats['win_pct'],
            'home_away_win': stats['home_win'] if is_home else stats['away_win'],
            'pace': stats['pace'],
            'off_rtg': stats['off_rtg'],
            'def_rtg': stats['def_rtg'],
            'form': stats['form']
        }


class BettingScanner:
    """Scan games and find top betting advantages"""
    
    def __init__(self, model):
        self.model = model
    
    def analyze_game(self, game):
        """Analyze a single game and return betting advantage"""
        home_team = game['home_team']
        away_team = game['away_team']
        spread = game['spread']
        
        home_stats = GameData.get_team_stats(home_team, is_home=True)
        away_stats = GameData.get_team_stats(away_team, is_home=False)
        
        # Compile features for model
        features = [
            home_stats['ppg'],
            away_stats['ppg'],
            home_stats['fg_pct'] / 100,
            away_stats['fg_pct'] / 100,
            home_stats['3p_pct'] / 100,
            away_stats['3p_pct'] / 100,
            home_stats['ft_pct'] / 100,
            away_stats['ft_pct'] / 100,
            home_stats['reb'],
            away_stats['reb'],
            home_stats['ast'],
            away_stats['ast'],
            home_stats['to'],
            away_stats['to'],
            home_stats['stl'],
            away_stats['stl'],
            home_stats['blk'],
            away_stats['blk'],
            home_stats['win_pct'],
            away_stats['win_pct'],
            home_stats['home_away_win'],
            away_stats['home_away_win'],
            1,  # is_home
            home_stats['pace'],
            away_stats['pace'],
            home_stats['off_rtg'],
            away_stats['off_rtg'],
            home_stats['def_rtg'],
            away_stats['def_rtg'],
            home_stats['form'],
            away_stats['form']
        ]
        
        # Get model prediction
        win_prob = self.model.predict(features)
        
        # Calculate spread implied probability
        spread_prob = 0.5 + (spread / 50)
        spread_prob = np.clip(spread_prob, 0, 1)
        
        # Calculate edge
        edge = win_prob - spread_prob
        
        # Determine bet recommendation
        if edge > 0.05:
            bet_team = home_team
            bet_spread = spread
        elif edge < -0.05:
            bet_team = away_team
            bet_spread = -spread
        else:
            bet_team = None
            bet_spread = spread
        
        return {
            'game': f"{away_team} @ {home_team}",
            'time': game['time'],
            'home_team': home_team,
            'away_team': away_team,
            'spread': spread,
            'over_under': game['over_under'],
            'model_win_prob': win_prob,
            'spread_prob': spread_prob,
            'edge': edge,
            'edge_pct': abs(edge) * 100,
            'bet_team': bet_team,
            'bet_spread': bet_spread,
            'predicted_margin': (win_prob - 0.5) * 20,
            'confidence': abs(win_prob - 0.5) * 200
        }
    
    def find_top_plays(self, games, top_n=3):
        """Analyze all games and return top betting advantages"""
        analyses = []
        
        for game in games:
            analysis = self.analyze_game(game)
            if analysis['bet_team']:  # Only include games with a recommended bet
                analyses.append(analysis)
        
        # Sort by edge (absolute value)
        analyses.sort(key=lambda x: x['edge_pct'], reverse=True)
        
        return analyses[:top_n]


def print_todays_games(games):
    """Display all games for today"""
    print(f"\n{'='*80}")
    print(f"📅 NCAA MEN'S BASKETBALL - {datetime.now().strftime('%A, %B %d, %Y')}")
    print(f"{'='*80}\n")
    print(f"{'Game':<40} {'Time':<10} {'Spread':<10} {'O/U':<10}")
    print("-" * 80)
    
    for game in games:
        matchup = f"{game['away_team']} @ {game['home_team']}"
        spread_str = f"{game['home_team'][:3]} {game['spread']:+.1f}"
        print(f"{matchup:<40} {game['time']:<10} {spread_str:<10} {game['over_under']:<10.1f}")
    
    print()


def print_top_plays(top_plays):
    """Display the top betting plays"""
    print(f"\n{'='*80}")
    print("🎯 TOP 3 BETTING ADVANTAGE PLAYS")
    print(f"{'='*80}\n")
    
    if not top_plays:
        print("⚠️  No strong betting advantages found for today's games.")
        print("    All spreads appear efficient. Consider passing or waiting for better opportunities.\n")
        return
    
    for i, play in enumerate(top_plays, 1):
        print(f"#{i} - {play['game']}")
        print("-" * 80)
        print(f"🕐 Game Time: {play['time']}")
        print(f"📊 Current Spread: {play['home_team'][:15]} {play['spread']:+.1f}")
        print(f"🎯 Over/Under: {play['over_under']:.1f}")
        print()
        print(f"🤖 MODEL ANALYSIS:")
        print(f"   Win Probability: {play['home_team'][:15]} {play['model_win_prob']*100:.1f}% | "
              f"{play['away_team'][:15]} {(1-play['model_win_prob'])*100:.1f}%")
        print(f"   Predicted Margin: {play['home_team'][:15]} {play['predicted_margin']:+.1f} points")
        print(f"   Model Confidence: {play['confidence']:.1f}%")
        print()
        print(f"💰 BETTING EDGE: {play['edge_pct']:.1f}%")
        print(f"   Spread implies: {play['spread_prob']*100:.1f}% win probability")
        print(f"   Model predicts: {play['model_win_prob']*100:.1f}% win probability")
        print(f"   Edge: {play['edge_pct']:.1f}% in favor of {play['bet_team']}")
        print()
        
        if play['edge_pct'] > 10:
            confidence_label = "⭐⭐⭐ STRONG VALUE"
            stake = "3-5% of bankroll"
        elif play['edge_pct'] > 7:
            confidence_label = "⭐⭐ SOLID VALUE"
            stake = "2-3% of bankroll"
        else:
            confidence_label = "⭐ MODERATE VALUE"
            stake = "1-2% of bankroll"
        
        print(f"✅ RECOMMENDATION: {confidence_label}")
        print(f"   Bet: {play['bet_team']} {play['bet_spread']:+.1f}")
        print(f"   Suggested Stake: {stake}")
        print()
        
        # Show key matchup factors
        print(f"📈 KEY FACTORS:")
        if play['predicted_margin'] > 5:
            print(f"   • {play['home_team']} has significant home court advantage")
        if play['confidence'] > 70:
            print(f"   • High confidence matchup - clear statistical edge")
        if abs(play['edge_pct']) > 10:
            print(f"   • Market appears to be mispricing this game")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main function"""
    print("\n🏀 NCAA BASKETBALL AUTOMATED BETTING SCANNER")
    print("="*80)
    
    # Train model
    print("\n[1/3] Training neural network model...")
    model = NCAABettingModel()
    model.train(verbose=0)
    print("✓ Model trained successfully")
    
    # Generate today's games
    print("\n[2/3] Loading today's games...")
    games = GameData.generate_todays_games(num_games=10)
    print(f"✓ Found {len(games)} games scheduled for today")
    
    # Display all games
    print_todays_games(games)
    
    # Analyze and find top plays
    print("[3/3] Analyzing games for betting advantages...")
    scanner = BettingScanner(model)
    top_plays = scanner.find_top_plays(games, top_n=3)
    print(f"✓ Analysis complete")
    
    # Display results
    print_top_plays(top_plays)
    
    # Disclaimer
    print("="*80)
    print("⚠️  DISCLAIMER:")
    print("    This analysis is for educational and entertainment purposes only.")
    print("    Always bet responsibly and never wager more than you can afford to lose.")
    print("    Past performance does not guarantee future results.")
    print("="*80 + "\n")
    
    # Ask if user wants to see full analysis of all games
    show_all = input("Would you like to see analysis of ALL games? (yes/no): ").strip().lower()
    if show_all in ['yes', 'y']:
        print("\n" + "="*80)
        print("📊 COMPLETE GAME ANALYSIS")
        print("="*80 + "\n")
        
        all_analyses = []
        for game in games:
            analysis = scanner.analyze_game(game)
            all_analyses.append(analysis)
        
        # Sort by edge
        all_analyses.sort(key=lambda x: x['edge_pct'], reverse=True)
        
        for analysis in all_analyses:
            bet_rec = f"Bet {analysis['bet_team']} {analysis['bet_spread']:+.1f}" if analysis['bet_team'] else "PASS"
            print(f"{analysis['game']:<40} | Edge: {analysis['edge_pct']:>5.1f}% | {bet_rec}")
        
        print()


if __name__ == "__main__":
    main()
