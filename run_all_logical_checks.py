"""
Unified Test Runner for All Logical Checks
Runs comprehensive checks on both intraday and daily stock forecasting
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from logical_checks_intraday import IntradayLogicalChecker
from logical_checks_daily import DailyStockLogicalChecker


class UnifiedLogicalCheckRunner:
    """Runs all logical checks and generates comprehensive reports."""

    def __init__(self, output_dir: str = "./check_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_intraday_checks(self, data_path: str = None, use_sample: bool = True) -> Dict[str, Any]:
        """Run intraday forecasting checks."""
        print("\n" + "="*80)
        print("RUNNING INTRADAY FORECASTING CHECKS")
        print("="*80 + "\n")

        try:
            from intraday_forecasting.predictor import IntradayPredictor

            # Load sample data if available
            if data_path and os.path.exists(data_path):
                df = pd.read_csv(data_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif use_sample:
                # Create sample data for testing
                print("Creating sample intraday data for checks...")
                df = self._create_sample_intraday_data()
            else:
                print("No data provided. Skipping intraday checks.")
                return {"status": "skipped", "reason": "no_data"}

            # Initialize predictor with minimal config
            predictor = IntradayPredictor(
                target_column='close',
                timeframe='5min',
                prediction_horizon=3,
                group_column='symbol',
                verbose=True
            )

            # Prepare data
            df_processed = predictor.prepare_features(df, fit_scaler=True)

            # Split data
            train_df, val_df, test_df = self._split_data(df_processed, 'timestamp', 'symbol')

            # IMPORTANT: Call prepare_data to initialize target scalers
            # This step fits the target scalers per group
            print("Fitting target scalers on training data...")
            try:
                X_train, y_train = predictor.prepare_data(train_df, fit_scaler=True)
                print(f"  Target scalers fitted: {len(predictor.group_target_scalers)} groups")
            except Exception as e:
                print(f"  Note: Could not fit target scalers (this is expected for small samples): {e}")

            # Initialize checker
            checker = IntradayLogicalChecker(verbose=True)

            # Run checks
            results = checker.run_all_checks(
                predictor=predictor,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                original_df=df,
                time_col='timestamp'
            )

            # Save results
            self._save_results(results, "intraday", checker)

            return {
                "status": "completed",
                "passed": checker.passed_checks,
                "failed": checker.failed_checks,
                "total": checker.passed_checks + checker.failed_checks,
                "pass_rate": (checker.passed_checks / (checker.passed_checks + checker.failed_checks) * 100)
                            if (checker.passed_checks + checker.failed_checks) > 0 else 0
            }

        except Exception as e:
            print(f"Error running intraday checks: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    def run_daily_checks(self, data_path: str = None, use_sample: bool = True) -> Dict[str, Any]:
        """Run daily stock forecasting checks."""
        print("\n" + "="*80)
        print("RUNNING DAILY STOCK FORECASTING CHECKS")
        print("="*80 + "\n")

        try:
            from daily_stock_forecasting.predictor import StockPredictor

            # Load sample data if available
            if data_path and os.path.exists(data_path):
                df = pd.read_csv(data_path)
                df['date'] = pd.to_datetime(df['date'])
            elif use_sample:
                # Create sample data for testing
                print("Creating sample daily stock data for checks...")
                df = self._create_sample_daily_data()
            else:
                print("No data provided. Skipping daily checks.")
                return {"status": "skipped", "reason": "no_data"}

            # Initialize predictor with minimal config
            predictor = StockPredictor(
                target_column='close',
                sequence_length=5,
                prediction_horizon=3,
                group_column='symbol',
                verbose=True
            )

            # Prepare data
            df_processed = predictor.prepare_features(df, fit_scaler=True)

            # Split data
            train_df, val_df, test_df = self._split_data(df_processed, 'date', 'symbol')

            # IMPORTANT: Call prepare_data to initialize target scalers
            # This step fits the target scalers per group
            print("Fitting target scalers on training data...")
            try:
                X_train, y_train = predictor.prepare_data(train_df, fit_scaler=True)
                print(f"  Target scalers fitted: {len(predictor.group_target_scalers)} groups")
            except Exception as e:
                print(f"  Note: Could not fit target scalers (this is expected for small samples): {e}")

            # Initialize checker
            checker = DailyStockLogicalChecker(verbose=True)

            # Run checks
            results = checker.run_all_checks(
                predictor=predictor,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                original_df=df,
                time_col='date'
            )

            # Save results
            self._save_results(results, "daily", checker)

            return {
                "status": "completed",
                "passed": checker.passed_checks,
                "failed": checker.failed_checks,
                "total": checker.passed_checks + checker.failed_checks,
                "pass_rate": (checker.passed_checks / (checker.passed_checks + checker.failed_checks) * 100)
                            if (checker.passed_checks + checker.failed_checks) > 0 else 0
            }

        except Exception as e:
            print(f"Error running daily checks: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    def _create_sample_intraday_data(self) -> pd.DataFrame:
        """Create sample intraday data for testing."""
        np.random.seed(42)

        symbols = ['AAPL', 'GOOGL', 'MSFT']
        data = []

        for symbol in symbols:
            # 200 5-minute bars per symbol (need > 96 + prediction_horizon for sequences)
            timestamps = pd.date_range('2024-01-01 09:30', periods=200, freq='5min')

            base_price = np.random.uniform(100, 500)
            prices = base_price + np.cumsum(np.random.randn(200) * 2)

            for i, ts in enumerate(timestamps):
                data.append({
                    'timestamp': ts,
                    'symbol': symbol,
                    'open': prices[i] + np.random.randn() * 0.5,
                    'high': prices[i] + abs(np.random.randn() * 1),
                    'low': prices[i] - abs(np.random.randn() * 1),
                    'close': prices[i],
                    'volume': np.random.randint(1000000, 10000000)
                })

        return pd.DataFrame(data)

    def _create_sample_daily_data(self) -> pd.DataFrame:
        """Create sample daily stock data for testing."""
        np.random.seed(42)

        symbols = ['AAPL', 'GOOGL', 'MSFT']
        data = []

        for symbol in symbols:
            # 100 days per symbol
            dates = pd.bdate_range('2024-01-01', periods=100)

            base_price = np.random.uniform(100, 500)
            prices = base_price + np.cumsum(np.random.randn(100) * 5)

            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'open': prices[i] + np.random.randn() * 2,
                    'high': prices[i] + abs(np.random.randn() * 3),
                    'low': prices[i] - abs(np.random.randn() * 3),
                    'close': prices[i],
                    'volume': np.random.randint(5000000, 50000000)
                })

        return pd.DataFrame(data)

    def _split_data(self, df: pd.DataFrame, time_col: str, group_col: str,
                    train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple:
        """Split data into train/val/test maintaining temporal order per group."""
        train_dfs = []
        val_dfs = []
        test_dfs = []

        for group in df[group_col].unique():
            group_df = df[df[group_col] == group].sort_values(time_col)

            n = len(group_df)
            train_size = int(n * train_ratio)
            val_size = int(n * val_ratio)

            train_dfs.append(group_df.iloc[:train_size])
            val_dfs.append(group_df.iloc[train_size:train_size + val_size])
            test_dfs.append(group_df.iloc[train_size + val_size:])

        return (
            pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame(),
            pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame(),
            pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
        )

    def _save_results(self, results: Dict, forecast_type: str, checker: Any):
        """Save check results to file."""
        # Save JSON results
        output_file = os.path.join(
            self.output_dir,
            f"{forecast_type}_checks_{self.timestamp}.json"
        )

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✅ Results saved to: {output_file}")

        # Save summary report
        summary_file = os.path.join(
            self.output_dir,
            f"{forecast_type}_summary_{self.timestamp}.txt"
        )

        with open(summary_file, 'w') as f:
            f.write(f"{forecast_type.upper()} FORECASTING - LOGICAL CHECKS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Checks: {checker.passed_checks + checker.failed_checks}\n")
            f.write(f"Passed: {checker.passed_checks}\n")
            f.write(f"Failed: {checker.failed_checks}\n")
            f.write(f"Pass Rate: {(checker.passed_checks / (checker.passed_checks + checker.failed_checks) * 100):.1f}%\n\n")

            if checker.failed_checks > 0:
                f.write("FAILED CHECKS:\n")
                f.write("-"*80 + "\n")
                for check_id, result in checker.check_results.items():
                    if not result['passed']:
                        f.write(f"\n{check_id}: {result['message']}\n")
                        if result.get('details'):
                            f.write(f"  Details: {result['details']}\n")

        print(f"✅ Summary saved to: {summary_file}")

    def generate_combined_report(self, intraday_results: Dict, daily_results: Dict):
        """Generate a combined report for both forecasting types."""
        report_file = os.path.join(
            self.output_dir,
            f"combined_report_{self.timestamp}.md"
        )

        with open(report_file, 'w') as f:
            f.write("# Comprehensive Logical Checks Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Intraday section
            f.write("## 🕒 Intraday Forecasting\n\n")
            if intraday_results.get('status') == 'completed':
                f.write(f"- **Total Checks:** {intraday_results['total']}\n")
                f.write(f"- **Passed:** {intraday_results['passed']} ✅\n")
                f.write(f"- **Failed:** {intraday_results['failed']} ❌\n")
                f.write(f"- **Pass Rate:** {intraday_results['pass_rate']:.1f}%\n\n")
            else:
                f.write(f"- **Status:** {intraday_results.get('status', 'unknown')}\n")
                if 'error' in intraday_results:
                    f.write(f"- **Error:** {intraday_results['error']}\n\n")

            # Daily section
            f.write("## 📅 Daily Stock Forecasting\n\n")
            if daily_results.get('status') == 'completed':
                f.write(f"- **Total Checks:** {daily_results['total']}\n")
                f.write(f"- **Passed:** {daily_results['passed']} ✅\n")
                f.write(f"- **Failed:** {daily_results['failed']} ❌\n")
                f.write(f"- **Pass Rate:** {daily_results['pass_rate']:.1f}%\n\n")
            else:
                f.write(f"- **Status:** {daily_results.get('status', 'unknown')}\n")
                if 'error' in daily_results:
                    f.write(f"- **Error:** {daily_results['error']}\n\n")

            # Overall summary
            f.write("## 📊 Overall Summary\n\n")
            if intraday_results.get('status') == 'completed' and daily_results.get('status') == 'completed':
                total_checks = intraday_results['total'] + daily_results['total']
                total_passed = intraday_results['passed'] + daily_results['passed']
                total_failed = intraday_results['failed'] + daily_results['failed']
                overall_pass_rate = (total_passed / total_checks * 100) if total_checks > 0 else 0

                f.write(f"- **Combined Total Checks:** {total_checks}\n")
                f.write(f"- **Combined Passed:** {total_passed}\n")
                f.write(f"- **Combined Failed:** {total_failed}\n")
                f.write(f"- **Overall Pass Rate:** {overall_pass_rate:.1f}%\n\n")

            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            f.write("Based on the checks performed, please review:\n\n")
            f.write("1. **Critical Checks (14.1.x):** These are priority checks that most affect model performance\n")
            f.write("2. **Scaling Issues (Section 2):** Ensure per-group scaling is correctly implemented\n")
            f.write("3. **Temporal Order (Section 1):** Verify data remains sorted throughout pipeline\n")
            f.write("4. **Data Leakage (Section 6):** Confirm no future information leaks into training\n\n")

            f.write("## 📁 Output Files\n\n")
            f.write(f"- Intraday JSON: `intraday_checks_{self.timestamp}.json`\n")
            f.write(f"- Intraday Summary: `intraday_summary_{self.timestamp}.txt`\n")
            f.write(f"- Daily JSON: `daily_checks_{self.timestamp}.json`\n")
            f.write(f"- Daily Summary: `daily_summary_{self.timestamp}.txt`\n")
            f.write(f"- This Report: `combined_report_{self.timestamp}.md`\n")

        print(f"\n✅ Combined report saved to: {report_file}")
        return report_file


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("UNIFIED LOGICAL CHECKS RUNNER")
    print("Comprehensive checks for Intraday and Daily Stock Forecasting")
    print("="*80 + "\n")

    runner = UnifiedLogicalCheckRunner()

    # Run intraday checks
    print("\n🚀 Starting Intraday Checks...")
    intraday_results = runner.run_intraday_checks(use_sample=True)

    # Run daily checks
    print("\n🚀 Starting Daily Stock Checks...")
    daily_results = runner.run_daily_checks(use_sample=True)

    # Generate combined report
    print("\n📝 Generating Combined Report...")
    report_file = runner.generate_combined_report(intraday_results, daily_results)

    # Final summary
    print("\n" + "="*80)
    print("ALL CHECKS COMPLETED")
    print("="*80)

    if intraday_results.get('status') == 'completed':
        print(f"\n✅ Intraday: {intraday_results['passed']}/{intraday_results['total']} checks passed "
              f"({intraday_results['pass_rate']:.1f}%)")

    if daily_results.get('status') == 'completed':
        print(f"✅ Daily: {daily_results['passed']}/{daily_results['total']} checks passed "
              f"({daily_results['pass_rate']:.1f}%)")

    print(f"\n📊 Combined report available at: {report_file}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
