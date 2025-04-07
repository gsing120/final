import unittest
import sys
import os
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all major system components
from backend.ai.gemma_quantitative_analysis import GemmaQuantitativeAnalyzer
from backend.indicators.trend_indicators import TrendIndicators
from backend.indicators.momentum_indicators import MomentumIndicators
from backend.indicators.volatility_indicators import VolatilityIndicators
from backend.indicators.volume_indicators import VolumeIndicators
from backend.indicators.cycle_indicators import CycleIndicators
from backend.indicators.pattern_recognition import PatternRecognition
from backend.indicators.custom_indicators import CustomIndicators

import numpy as np
import pandas as pd

class TestIndicatorLibrary(unittest.TestCase):
    def setUp(self):
        # Create sample market data for testing
        np.random.seed(42)  # For reproducibility
        self.dates = pd.date_range(start='2025-01-01', periods=100)
        
        # Create price data
        base_price = 100
        # Create a price series with a trend, cycle, and noise
        trend = np.linspace(0, 20, 100)  # Upward trend
        cycle = 10 * np.sin(np.linspace(0, 4*np.pi, 100))  # Cyclical component
        noise = np.random.normal(0, 5, 100)  # Random noise
        
        self.prices = base_price + trend + cycle + noise
        self.volumes = np.random.normal(1000000, 200000, 100)
        
        # Add volume spikes
        self.volumes[30:35] = self.volumes[30:35] * 3  # Volume spike
        self.volumes[70:75] = self.volumes[70:75] * 2.5  # Another volume spike
        
        # Create OHLC data
        self.opens = self.prices - np.random.normal(0, 1, 100)
        self.highs = np.maximum(self.prices, self.opens) + np.random.normal(0.5, 0.5, 100)
        self.lows = np.minimum(self.prices, self.opens) - np.random.normal(0.5, 0.5, 100)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'open': self.opens,
            'high': self.highs,
            'low': self.lows,
            'close': self.prices,
            'volume': self.volumes
        }, index=self.dates)
        
        # Initialize indicator libraries
        self.trend_indicators = TrendIndicators()
        self.momentum_indicators = MomentumIndicators()
        self.volatility_indicators = VolatilityIndicators()
        self.volume_indicators = VolumeIndicators()
        self.cycle_indicators = CycleIndicators()
        self.pattern_recognition = PatternRecognition()
        self.custom_indicators = CustomIndicators()
        
    def test_trend_indicators(self):
        """Test all trend indicators"""
        # Test SMA
        sma = self.trend_indicators.sma(self.data['close'], 20)
        self.assertEqual(len(sma), len(self.data))
        self.assertTrue(pd.isna(sma.iloc[0]))  # First values should be NaN
        self.assertFalse(pd.isna(sma.iloc[20]))  # Value at period should be available
        
        # Test EMA
        ema = self.trend_indicators.ema(self.data['close'], 20)
        self.assertEqual(len(ema), len(self.data))
        self.assertTrue(pd.isna(ema.iloc[0]))
        self.assertFalse(pd.isna(ema.iloc[20]))
        
        # Test MACD
        macd, signal, hist = self.trend_indicators.macd(self.data['close'])
        self.assertEqual(len(macd), len(self.data))
        self.assertEqual(len(signal), len(self.data))
        self.assertEqual(len(hist), len(self.data))
        
        # Test Bollinger Bands
        upper, middle, lower = self.trend_indicators.bollinger_bands(self.data['close'])
        self.assertEqual(len(upper), len(self.data))
        self.assertEqual(len(middle), len(self.data))
        self.assertEqual(len(lower), len(self.data))
        
        # Test Parabolic SAR
        psar = self.trend_indicators.parabolic_sar(self.data['high'], self.data['low'])
        self.assertEqual(len(psar), len(self.data))
        
        # Test Ichimoku Cloud
        tenkan, kijun, senkou_a, senkou_b, chikou = self.trend_indicators.ichimoku_cloud(
            self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(tenkan), len(self.data))
        self.assertEqual(len(kijun), len(self.data))
        self.assertEqual(len(senkou_a), len(self.data))
        self.assertEqual(len(senkou_b), len(self.data))
        self.assertEqual(len(chikou), len(self.data))
        
        # Test ADX
        adx = self.trend_indicators.adx(self.data['high'], self.data['low'], self.data['close'])
        self.assertEqual(len(adx), len(self.data))
        
        # Test Supertrend
        supertrend, direction = self.trend_indicators.supertrend(
            self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(supertrend), len(self.data))
        self.assertEqual(len(direction), len(self.data))
        
    def test_momentum_indicators(self):
        """Test all momentum indicators"""
        # Test RSI
        rsi = self.momentum_indicators.rsi(self.data['close'])
        self.assertEqual(len(rsi), len(self.data))
        self.assertTrue(all(0 <= x <= 100 for x in rsi.dropna()))  # RSI should be between 0 and 100
        
        # Test Stochastic Oscillator
        k, d = self.momentum_indicators.stochastic(
            self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(k), len(self.data))
        self.assertEqual(len(d), len(self.data))
        self.assertTrue(all(0 <= x <= 100 for x in k.dropna()))
        self.assertTrue(all(0 <= x <= 100 for x in d.dropna()))
        
        # Test CCI
        cci = self.momentum_indicators.cci(
            self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(cci), len(self.data))
        
        # Test Williams %R
        williams_r = self.momentum_indicators.williams_r(
            self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(williams_r), len(self.data))
        self.assertTrue(all(-100 <= x <= 0 for x in williams_r.dropna()))
        
        # Test ROC
        roc = self.momentum_indicators.rate_of_change(self.data['close'])
        self.assertEqual(len(roc), len(self.data))
        
        # Test Awesome Oscillator
        ao = self.momentum_indicators.awesome_oscillator(
            self.data['high'], self.data['low']
        )
        self.assertEqual(len(ao), len(self.data))
        
        # Test Money Flow Index
        mfi = self.momentum_indicators.money_flow_index(
            self.data['high'], self.data['low'], self.data['close'], self.data['volume']
        )
        self.assertEqual(len(mfi), len(self.data))
        self.assertTrue(all(0 <= x <= 100 for x in mfi.dropna()))
        
        # Test TSI
        tsi = self.momentum_indicators.true_strength_index(self.data['close'])
        self.assertEqual(len(tsi), len(self.data))
        
    def test_volatility_indicators(self):
        """Test all volatility indicators"""
        # Test ATR
        atr = self.volatility_indicators.atr(
            self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(atr), len(self.data))
        self.assertTrue(all(x >= 0 for x in atr.dropna()))  # ATR should be positive
        
        # Test Standard Deviation
        std = self.volatility_indicators.standard_deviation(self.data['close'])
        self.assertEqual(len(std), len(self.data))
        self.assertTrue(all(x >= 0 for x in std.dropna()))
        
        # Test Keltner Channel
        upper, middle, lower = self.volatility_indicators.keltner_channel(
            self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(upper), len(self.data))
        self.assertEqual(len(middle), len(self.data))
        self.assertEqual(len(lower), len(self.data))
        
        # Test Historical Volatility
        hv = self.volatility_indicators.historical_volatility(self.data['close'])
        self.assertEqual(len(hv), len(self.data))
        self.assertTrue(all(x >= 0 for x in hv.dropna()))
        
        # Test Ulcer Index
        ui = self.volatility_indicators.ulcer_index(self.data['close'])
        self.assertEqual(len(ui), len(self.data))
        self.assertTrue(all(x >= 0 for x in ui.dropna()))
        
        # Test Normalized Average True Range
        natr = self.volatility_indicators.normalized_atr(
            self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(natr), len(self.data))
        
        # Test Chaikin Volatility
        chaikin_vol = self.volatility_indicators.chaikin_volatility(
            self.data['high'], self.data['low']
        )
        self.assertEqual(len(chaikin_vol), len(self.data))
        
    def test_volume_indicators(self):
        """Test all volume indicators"""
        # Test OBV
        obv = self.volume_indicators.on_balance_volume(
            self.data['close'], self.data['volume']
        )
        self.assertEqual(len(obv), len(self.data))
        
        # Test VWAP
        vwap = self.volume_indicators.vwap(
            self.data['high'], self.data['low'], self.data['close'], self.data['volume']
        )
        self.assertEqual(len(vwap), len(self.data))
        
        # Test Accumulation/Distribution Line
        adl = self.volume_indicators.accumulation_distribution_line(
            self.data['high'], self.data['low'], self.data['close'], self.data['volume']
        )
        self.assertEqual(len(adl), len(self.data))
        
        # Test Chaikin Money Flow
        cmf = self.volume_indicators.chaikin_money_flow(
            self.data['high'], self.data['low'], self.data['close'], self.data['volume']
        )
        self.assertEqual(len(cmf), len(self.data))
        
        # Test Force Index
        fi = self.volume_indicators.force_index(
            self.data['close'], self.data['volume']
        )
        self.assertEqual(len(fi), len(self.data))
        
        # Test Ease of Movement
        eom = self.volume_indicators.ease_of_movement(
            self.data['high'], self.data['low'], self.data['volume']
        )
        self.assertEqual(len(eom), len(self.data))
        
        # Test Volume Profile
        volume_profile = self.volume_indicators.volume_profile(
            self.data['close'], self.data['volume'], num_bins=10
        )
        self.assertEqual(len(volume_profile), 10)  # Should have 10 bins
        
        # Test Volume Weighted MACD
        vw_macd, vw_signal, vw_hist = self.volume_indicators.volume_weighted_macd(
            self.data['close'], self.data['volume']
        )
        self.assertEqual(len(vw_macd), len(self.data))
        self.assertEqual(len(vw_signal), len(self.data))
        self.assertEqual(len(vw_hist), len(self.data))
        
    def test_cycle_indicators(self):
        """Test all cycle indicators"""
        # Test Hilbert Transform - Dominant Cycle Period
        dcp = self.cycle_indicators.dominant_cycle_period(self.data['close'])
        self.assertEqual(len(dcp), len(self.data))
        
        # Test Hilbert Transform - Sine Wave
        sine, lead_sine = self.cycle_indicators.ht_sine_wave(self.data['close'])
        self.assertEqual(len(sine), len(self.data))
        self.assertEqual(len(lead_sine), len(self.data))
        
        # Test Hilbert Transform - Trend vs Cycle Mode
        trend_mode = self.cycle_indicators.ht_trend_mode(self.data['close'])
        self.assertEqual(len(trend_mode), len(self.data))
        
        # Test Mesa Sine Wave
        sine_wave, lead_sine = self.cycle_indicators.mesa_sine_wave(self.data['close'])
        self.assertEqual(len(sine_wave), len(self.data))
        self.assertEqual(len(lead_sine), len(self.data))
        
        # Test Ehlers Fisher Transform
        fisher, trigger = self.cycle_indicators.ehlers_fisher_transform(self.data['close'])
        self.assertEqual(len(fisher), len(self.data))
        self.assertEqual(len(trigger), len(self.data))
        
    def test_pattern_recognition(self):
        """Test pattern recognition indicators"""
        # Test Candlestick Patterns
        doji = self.pattern_recognition.doji(
            self.data['open'], self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(doji), len(self.data))
        
        hammer = self.pattern_recognition.hammer(
            self.data['open'], self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(hammer), len(self.data))
        
        engulfing = self.pattern_recognition.engulfing(
            self.data['open'], self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(engulfing), len(self.data))
        
        morning_star = self.pattern_recognition.morning_star(
            self.data['open'], self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(morning_star), len(self.data))
        
        # Test Chart Patterns
        head_shoulders = self.pattern_recognition.head_and_shoulders(
            self.data['close']
        )
        self.assertEqual(len(head_shoulders), len(self.data))
        
        double_top = self.pattern_recognition.double_top(
            self.data['close']
        )
        self.assertEqual(len(double_top), len(self.data))
        
        triangle = self.pattern_recognition.triangle(
            self.data['high'], self.data['low']
        )
        self.assertEqual(len(triangle), len(self.data))
        
        # Test Support/Resistance
        support, resistance = self.pattern_recognition.support_resistance(
            self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(support), len(self.data))
        self.assertEqual(len(resistance), len(self.data))
        
    def test_custom_indicators(self):
        """Test custom indicators"""
        # Test Heikin-Ashi
        ha_open, ha_high, ha_low, ha_close = self.custom_indicators.heikin_ashi(
            self.data['open'], self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(ha_open), len(self.data))
        self.assertEqual(len(ha_high), len(self.data))
        self.assertEqual(len(ha_low), len(self.data))
        self.assertEqual(len(ha_close), len(self.data))
        
        # Test Elder Ray
        bull_power, bear_power = self.custom_indicators.elder_ray(
            self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(bull_power), len(self.data))
        self.assertEqual(len(bear_power), len(self.data))
        
        # Test Klinger Volume Oscillator
        kvo, signal = self.custom_indicators.klinger_oscillator(
            self.data['high'], self.data['low'], self.data['close'], self.data['volume']
        )
        self.assertEqual(len(kvo), len(self.data))
        self.assertEqual(len(signal), len(self.data))
        
        # Test Relative Vigor Index
        rvi, signal = self.custom_indicators.relative_vigor_index(
            self.data['open'], self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(rvi), len(self.data))
        self.assertEqual(len(signal), len(self.data))
        
        # Test Squeeze Momentum
        squeeze, momentum = self.custom_indicators.squeeze_momentum(
            self.data['high'], self.data['low'], self.data['close']
        )
        self.assertEqual(len(squeeze), len(self.data))
        self.assertEqual(len(momentum), len(self.data))
        
        # Test Volume Zone Oscillator
        vzo = self.custom_indicators.volume_zone_oscillator(
            self.data['close'], self.data['volume']
        )
        self.assertEqual(len(vzo), len(self.data))
        
    def test_gemma_quantitative_analysis(self):
        """Test Gemma quantitative analysis integration with indicators"""
        # Initialize Gemma analyzer
        gemma_analyzer = GemmaQuantitativeAnalyzer()
        
        # Calculate indicators
        sma_20 = self.trend_indicators.sma(self.data['close'], 20)
        rsi_14 = self.momentum_indicators.rsi(self.data['close'])
        atr_14 = self.volatility_indicators.atr(
            self.data['high'], self.data['low'], self.data['close']
        )
        obv = self.volume_indicators.on_balance_volume(
            self.data['close'], self.data['volume']
        )
        
        # Combine indicators into a DataFrame
        indicators_df = pd.DataFrame({
            'close': self.data['close'],
            'sma_20': sma_20,
            'rsi_14': rsi_14,
            'atr_14': atr_14,
            'obv': obv
        })
        
        # Run Gemma analysis on indicators
        analysis_results = gemma_analyzer.analyze_indicators(indicators_df)
        
        # Check analysis results structure
        self.assertIsInstance(analysis_results, dict)
        self.assertIn('trend_analysis', analysis_results)
        self.assertIn('momentum_analysis', analysis_results)
        self.assertIn('volatility_analysis', analysis_results)
        self.assertIn('volume_analysis', analysis_results)
        self.assertIn('combined_signal', analysis_results)
        
        # Check that combined signal is a valid value
        combined_signal = analysis_results['combined_signal']
        self.assertIn(combined_signal, ['strong_buy', 'buy', 'neutral', 'sell', 'strong_sell'])
        
        # Test pattern detection
        pattern_results = gemma_analyzer.detect_patterns(self.data)
        
        # Check pattern results structure
        self.assertIsInstance(pattern_results, dict)
        self.assertIn('detected_patterns', pattern_results)
        self.assertIn('confidence_scores', pattern_results)
        self.assertIn('price_targets', pattern_results)
        
        # Test statistical analysis
        stats_results = gemma_analyzer.statistical_analysis(self.data['close'])
        
        # Check statistical results structure
        self.assertIsInstance(stats_results, dict)
        self.assertIn('mean', stats_results)
        self.assertIn('std_dev', stats_results)
        self.assertIn('skewness', stats_results)
        self.assertIn('kurtosis', stats_results)
        self.assertIn('normality_test', stats_results)
        
        # Test correlation analysis
        # Create a second price series with some correlation
        second_prices = self.prices * 0.7 + np.random.normal(0, 10, 100)
        correlation = gemma_analyzer.correlation_analysis(self.data['close'], pd.Series(second_prices))
        
        # Check correlation results
        self.assertIsInstance(correlation, dict)
        self.assertIn('pearson', correlation)
        self.assertIn('spearman', correlation)
        self.assertIn('kendall', correlation)
        self.assertIn('rolling_correlation', correlation)
        
        # Test regime detection
        regime_results = gemma_analyzer.detect_market_regime(self.data)
        
        # Check regime results
        self.assertIsInstance(regime_results, dict)
        self.assertIn('current_regime', regime_results)
        self.assertIn('regime_probabilities', regime_results)
        self.assertIn('regime_history', regime_results)
        
        # Test strategy recommendation
        strategy_recommendation = gemma_analyzer.recommend_strategy(self.data, analysis_results, regime_results)
        
        # Check strategy recommendation
        self.assertIsInstance(strategy_recommendation, dict)
        self.assertIn('recommended_strategy', strategy_recommendation)
        self.assertIn('rationale', strategy_recommendation)
        self.assertIn('parameters', strategy_recommendation)
        self.assertIn('expected_performance', strategy_recommendation)

if __name__ == '__main__':
    unittest.main()
