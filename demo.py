"""
Demo script showing the NCAA Basketball Betting Model in action
Uses example game data to demonstrate the model's capabilities
"""

import numpy as np
from ncaa_betting_model_sklearn import NCAABettingModel

def run_demo():
    """
    Run a demonstration with pre-filled game data
    """
    print("\n🏀 NCAA MEN'S BASKETBALL BETTING MODEL - DEMO")
    print("=" * 60)
    print("\nThis demo will show the model analyzing a sample matchup:")
    print("Duke Blue Devils vs North Carolina Tar Heels")
    print("=" * 60)
    
    # Initialize and train model
    print("\n[1/3] Training the neural network model...")
    model = NCAABettingModel()
    model.train(verbose=0)
    
    # Example game: Duke vs North Carolina
    print("\n[2/3] Loading sample game data...\n")
    
    # Duke's statistics (strong offensive team)
    duke_stats = {
        'PPG': 82.5,
        'FG%': 47.2,
        '3P%': 36.8,
        'FT%': 75.4,
        'REB': 38.2,
        'AST': 16.3,
        'TO': 11.8,
        'STL': 7.5,
        'BLK': 4.8,
        'WIN%': 78.0,
        'HOME_WIN%': 88.0,
        'RECENT_FORM': 4  # 4 wins in last 5
    }
    
    # UNC's statistics (balanced team)
    unc_stats = {
        'PPG': 79.8,
        'FG%': 45.9,
        '3P%': 34.2,
        'FT%': 71.8,
        'REB': 39.5,
        'AST': 15.1,
        'TO': 12.5,
        'STL': 6.8,
        'BLK': 3.9,
        'WIN%': 72.0,
        'AWAY_WIN%': 65.0,
        'RECENT_FORM': 3  # 3 wins in last 5
    }
    
    print("DUKE BLUE DEVILS:")
    print(f"  PPG: {duke_stats['PPG']:.1f} | FG%: {duke_stats['FG%']:.1f} | 3P%: {duke_stats['3P%']:.1f}")
    print(f"  REB: {duke_stats['REB']:.1f} | AST: {duke_stats['AST']:.1f} | TO: {duke_stats['TO']:.1f}")
    print(f"  WIN%: {duke_stats['WIN%']:.1f} | Recent Form: {duke_stats['RECENT_FORM']}/5 wins")
    
    print("\nNORTH CAROLINA TAR HEELS:")
    print(f"  PPG: {unc_stats['PPG']:.1f} | FG%: {unc_stats['FG%']:.1f} | 3P%: {unc_stats['3P%']:.1f}")
    print(f"  REB: {unc_stats['REB']:.1f} | AST: {unc_stats['AST']:.1f} | TO: {unc_stats['TO']:.1f}")
    print(f"  WIN%: {unc_stats['WIN%']:.1f} | Recent Form: {unc_stats['RECENT_FORM']}/5 wins")
    
    print("\nGAME DETAILS:")
    print("  Location: Cameron Indoor Stadium (Duke home)")
    print("  Point Spread: Duke -4.5")
    print("  Over/Under: 160.5")
    
    # Compile features for the model
    team_stats = [
        duke_stats['PPG'],
        unc_stats['PPG'],
        duke_stats['FG%'] / 100,
        unc_stats['FG%'] / 100,
        duke_stats['3P%'] / 100,
        unc_stats['3P%'] / 100,
        duke_stats['FT%'] / 100,
        unc_stats['FT%'] / 100,
        duke_stats['REB'],
        unc_stats['REB'],
        duke_stats['AST'],
        unc_stats['AST'],
        duke_stats['TO'],
        unc_stats['TO'],
        duke_stats['STL'],
        unc_stats['STL'],
        duke_stats['BLK'],
        unc_stats['BLK'],
        duke_stats['WIN%'] / 100,
        unc_stats['WIN%'] / 100,
        duke_stats['HOME_WIN%'] / 100,
        unc_stats['AWAY_WIN%'] / 100,
        1,  # Duke is home
        70,  # Duke pace
        72,  # UNC pace
        112,  # Duke offensive rating
        108,  # UNC offensive rating
        98,  # Duke defensive rating
        102,  # UNC defensive rating
        duke_stats['RECENT_FORM'] / 5,
        unc_stats['RECENT_FORM'] / 5
    ]
    
    # Get predictions
    print("\n[3/3] Generating betting insights...\n")
    insights = model.get_betting_insights(
        team_stats,
        spread=-4.5,
        over_under=160.5,
        team_name="Duke",
        opponent_name="North Carolina"
    )
    
    # Additional analysis
    print("ADDITIONAL ANALYSIS:")
    print("-" * 60)
    print("\nKey Factors:")
    if insights['win_probability'] > 0.6:
        print("  ✓ Duke has a significant statistical advantage")
        print("  ✓ Home court advantage at Cameron Indoor is crucial")
        print("  ✓ Better recent form favors Duke")
    else:
        print("  ⚠ This is expected to be a competitive matchup")
        print("  ⚠ Home court may be the deciding factor")
    
    print(f"\nExpected Score: Duke {82.5 + insights['predicted_margin']/2:.0f}, "
          f"UNC {79.8 - insights['predicted_margin']/2:.0f}")
    
    print("\nBetting Strategy:")
    if abs(insights['spread_value']) > 0.05:
        print("  💰 This game offers value based on the model")
        print("  💰 Consider a moderate stake (2-3% of bankroll)")
    else:
        print("  ⚠ Spread appears efficient - limited value")
        print("  ⚠ Consider passing or making a small recreational bet")
    
    print("\n" + "=" * 60)
    print("Demo complete! Run ncaa_betting_model_sklearn.py for live input.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
