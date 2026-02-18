"""
Credit Risk ML — Business Threshold Optimization
==================================================
Cost-sensitive decision threshold selection for loan default prediction.

WHY NOT JUST USE 0.5?
─────────────────────
The default 0.5 threshold assumes symmetric misclassification costs.
In credit risk, costs are deeply asymmetric:
  - Missing a defaulter (FN) costs ~₹100K (loan write-off)
  - Rejecting a good customer (FP) costs ~₹10K (lost interest revenue)

This 10:1 asymmetry means we should lower the threshold to catch more
defaulters, even at the expense of rejecting some good customers.

Interview Insight:
"How do you choose the classification threshold?"
→ "I define a business cost matrix with the P&L team, sweep thresholds
   from 0.05 to 0.95, and select the threshold that minimizes expected
   total business loss. I also plot precision-recall vs threshold so
   stakeholders can visualize the trade-off."
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ─── Paths ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. COST MATRIX
# ============================================================

@dataclass
class CostMatrix:
    """
    Business cost matrix for credit risk decisions.

    WHY THESE VALUES?
    ─────────────────
    - FN (missed defaulter): Average loan size * expected loss-given-default.
      For unsecured personal loans in India, LGD is ~60-80%.
      ₹100K represents average exposure on a ₹150K loan at 67% LGD.
    - FP (rejected good customer): Lost interest income over loan tenure.
      ₹10K represents ~2 years of net interest margin on a ₹150K loan.
    - TP/TN: No incremental cost (correct decisions).

    In production, these are calibrated quarterly with the CFO.
    """
    fn_cost: float = 100_000   # Missed defaulter → loan write-off
    fp_cost: float = 10_000    # Rejected good customer → lost revenue
    tp_cost: float = 0         # Correctly caught default
    tn_cost: float = 0         # Correctly approved good customer

    @property
    def asymmetry_ratio(self) -> float:
        """FN/FP cost ratio. Higher = more aggressive at catching defaults."""
        return self.fn_cost / self.fp_cost if self.fp_cost > 0 else float('inf')

    def compute_total_cost(self, cm: np.ndarray) -> float:
        """
        Compute total business cost from a confusion matrix.

        Parameters
        ----------
        cm : np.ndarray, shape (2, 2)
            Confusion matrix: [[TN, FP], [FN, TP]]
        """
        tn, fp, fn, tp = cm.ravel()
        return (
            tn * self.tn_cost +
            fp * self.fp_cost +
            fn * self.fn_cost +
            tp * self.tp_cost
        )


# ============================================================
# 2. THRESHOLD OPTIMIZER
# ============================================================

@dataclass
class ThresholdResult:
    """Result of threshold analysis at a single point."""
    threshold: float
    total_cost: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int


class ThresholdOptimizer:
    """
    Cost-sensitive threshold optimizer.

    Sweeps through candidate thresholds and finds the one that
    minimizes total expected business loss.

    Usage:
        optimizer = ThresholdOptimizer(cost_matrix=CostMatrix())
        results = optimizer.optimize(y_true, y_proba)
        optimizer.plot_analysis(save_path='results/threshold_analysis.png')
        optimizer.print_comparison()
    """

    def __init__(self, cost_matrix: CostMatrix = None):
        self.cost_matrix = cost_matrix or CostMatrix()
        self.results: list[ThresholdResult] = []
        self.optimal: ThresholdResult | None = None
        self.default_result: ThresholdResult | None = None

    def _evaluate_threshold(self, y_true: np.ndarray,
                            y_proba: np.ndarray,
                            threshold: float) -> ThresholdResult:
        """Evaluate a single threshold."""
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Handle edge cases where all predictions are same class
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return ThresholdResult(
            threshold=threshold,
            total_cost=self.cost_matrix.compute_total_cost(cm),
            precision=prec,
            recall=rec,
            f1=f1,
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        )

    def optimize(self, y_true: np.ndarray, y_proba: np.ndarray,
                 thresholds: np.ndarray = None) -> ThresholdResult:
        """
        Find optimal threshold minimizing business cost.

        Parameters
        ----------
        y_true : array-like
            True binary labels (0/1).
        y_proba : array-like
            Predicted probabilities for the positive class.
        thresholds : array-like, optional
            Candidate thresholds. Default: np.arange(0.05, 0.96, 0.01).

        Returns
        -------
        ThresholdResult for the optimal threshold.
        """
        if thresholds is None:
            thresholds = np.arange(0.05, 0.96, 0.01)

        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        # Sweep all thresholds
        self.results = [
            self._evaluate_threshold(y_true, y_proba, t)
            for t in thresholds
        ]

        # Find minimum cost
        self.optimal = min(self.results, key=lambda r: r.total_cost)

        # Also compute default 0.5 for comparison
        self.default_result = self._evaluate_threshold(y_true, y_proba, 0.5)

        return self.optimal

    def plot_analysis(self, save_path: str = None):
        """
        Generate 3-panel analysis plot.

        Panel 1: Threshold vs Total Business Cost (₹)
        Panel 2: Threshold vs Recall (sensitivity)
        Panel 3: Threshold vs Precision

        WHY THESE 3?
        - Cost: The objective function we're optimizing
        - Recall: How many defaulters we catch (business cares most)
        - Precision: How many flagged customers are actually defaulters
        """
        if not self.results:
            raise ValueError("Run optimize() first.")

        thresholds = [r.threshold for r in self.results]
        costs = [r.total_cost for r in self.results]
        recalls = [r.recall for r in self.results]
        precisions = [r.precision for r in self.results]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Business Threshold Optimization Analysis',
                     fontsize=14, fontweight='bold', y=1.02)

        # ── Panel 1: Cost ──
        ax1 = axes[0]
        ax1.plot(thresholds, [c / 1e6 for c in costs],
                 color='#ef4444', linewidth=2)
        ax1.axvline(self.optimal.threshold, color='#10b981',
                    linestyle='--', linewidth=1.5,
                    label=f'Optimal: {self.optimal.threshold:.2f}')
        ax1.axvline(0.5, color='#6b7280', linestyle=':',
                    linewidth=1.5, label='Default: 0.50')
        ax1.set_xlabel('Decision Threshold')
        ax1.set_ylabel('Total Business Cost (₹ Millions)')
        ax1.set_title('Threshold vs Cost')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)

        # ── Panel 2: Recall ──
        ax2 = axes[1]
        ax2.plot(thresholds, recalls, color='#3b82f6', linewidth=2)
        ax2.axvline(self.optimal.threshold, color='#10b981',
                    linestyle='--', linewidth=1.5)
        ax2.axvline(0.5, color='#6b7280', linestyle=':', linewidth=1.5)
        ax2.set_xlabel('Decision Threshold')
        ax2.set_ylabel('Recall (Sensitivity)')
        ax2.set_title('Threshold vs Recall')
        ax2.grid(alpha=0.3)

        # ── Panel 3: Precision ──
        ax3 = axes[2]
        ax3.plot(thresholds, precisions, color='#f59e0b', linewidth=2)
        ax3.axvline(self.optimal.threshold, color='#10b981',
                    linestyle='--', linewidth=1.5)
        ax3.axvline(0.5, color='#6b7280', linestyle=':', linewidth=1.5)
        ax3.set_xlabel('Decision Threshold')
        ax3.set_ylabel('Precision')
        ax3.set_title('Threshold vs Precision')
        ax3.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor='white')
            print(f"  [OK] Plot saved: {save_path}")
        plt.close()

    def get_business_comparison(self) -> dict:
        """
        Side-by-side comparison: default 0.5 vs optimal threshold.

        Returns dict with metrics for both thresholds.
        """
        if not self.optimal or not self.default_result:
            raise ValueError("Run optimize() first.")

        d = self.default_result
        o = self.optimal

        savings = d.total_cost - o.total_cost
        savings_pct = (savings / d.total_cost * 100) if d.total_cost > 0 else 0

        return {
            'default_threshold': {
                'threshold': d.threshold,
                'total_cost': d.total_cost,
                'precision': d.precision,
                'recall': d.recall,
                'f1': d.f1,
                'fn_count': d.fn,
                'fp_count': d.fp,
            },
            'optimal_threshold': {
                'threshold': o.threshold,
                'total_cost': o.total_cost,
                'precision': o.precision,
                'recall': o.recall,
                'f1': o.f1,
                'fn_count': o.fn,
                'fp_count': o.fp,
            },
            'savings': savings,
            'savings_pct': savings_pct,
        }

    def print_comparison(self):
        """Print formatted comparison table."""
        comp = self.get_business_comparison()
        d = comp['default_threshold']
        o = comp['optimal_threshold']

        print("\n" + "=" * 65)
        print("  BUSINESS THRESHOLD OPTIMIZATION — RESULTS")
        print("=" * 65)
        print(f"\n  Cost Matrix:")
        print(f"    Missed Defaulter (FN):     ₹{self.cost_matrix.fn_cost:>10,.0f}")
        print(f"    Rejected Good Cust (FP):   ₹{self.cost_matrix.fp_cost:>10,.0f}")
        print(f"    Asymmetry Ratio:           {self.cost_matrix.asymmetry_ratio:.0f}:1")

        print(f"\n  {'Metric':<28} {'Default (0.5)':>14} {'Optimal':>14}")
        print(f"  {'─' * 56}")
        print(f"  {'Threshold':<28} {d['threshold']:>14.2f} {o['threshold']:>14.2f}")
        print(f"  {'Total Business Cost (₹)':<28} {d['total_cost']:>14,.0f} {o['total_cost']:>14,.0f}")
        print(f"  {'Precision':<28} {d['precision']:>14.4f} {o['precision']:>14.4f}")
        print(f"  {'Recall':<28} {d['recall']:>14.4f} {o['recall']:>14.4f}")
        print(f"  {'F1 Score':<28} {d['f1']:>14.4f} {o['f1']:>14.4f}")
        print(f"  {'Missed Defaulters (FN)':<28} {d['fn_count']:>14,} {o['fn_count']:>14,}")
        print(f"  {'Rejected Good Cust (FP)':<28} {d['fp_count']:>14,} {o['fp_count']:>14,}")

        print(f"\n  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  💰 SAVINGS: ₹{comp['savings']:>12,.0f} ({comp['savings_pct']:.1f}% reduction)    │")
        print(f"  └─────────────────────────────────────────────────────┘")

    def print_confusion_matrix(self, label: str = "Optimal"):
        """Print formatted confusion matrix at optimal threshold."""
        r = self.optimal if label == "Optimal" else self.default_result
        if not r:
            raise ValueError("Run optimize() first.")

        print(f"\n  Confusion Matrix @ threshold={r.threshold:.2f} ({label})")
        print(f"  {'─' * 40}")
        print(f"                    Predicted")
        print(f"                    Non-Def  Default")
        print(f"  Actual Non-Def    {r.tn:>6}   {r.fp:>6}")
        print(f"  Actual Default    {r.fn:>6}   {r.tp:>6}")


# ============================================================
# 3. MAIN EXECUTION
# ============================================================

def main():
    """Run threshold optimization on the best model (LightGBM)."""

    print("\n" + "=" * 65)
    print("  BUSINESS THRESHOLD OPTIMIZATION")
    print("=" * 65)

    # ── Load data and model ──
    print("\n  Loading test data and LightGBM model...")
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    model = joblib.load(MODELS_DIR / 'LightGBM_model.pkl')

    # Get predicted probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"    -> Test samples: {len(y_test):,}")
    print(f"    -> Default rate: {y_test.mean():.1%}")
    print(f"    -> Prob range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")

    # ── Optimize ──
    print("\n  Sweeping thresholds 0.05 → 0.95...")
    cost_matrix = CostMatrix(fn_cost=100_000, fp_cost=10_000)
    optimizer = ThresholdOptimizer(cost_matrix=cost_matrix)
    optimal = optimizer.optimize(y_test, y_proba)

    print(f"    -> Optimal threshold: {optimal.threshold:.2f}")
    print(f"    -> Minimum cost: ₹{optimal.total_cost:,.0f}")

    # ── Plot ──
    plot_path = RESULTS_DIR / 'threshold_analysis.png'
    optimizer.plot_analysis(save_path=str(plot_path))

    # ── Print results ──
    optimizer.print_comparison()
    optimizer.print_confusion_matrix("Optimal")
    optimizer.print_confusion_matrix("Default")

    # ── Save results JSON ──
    import json
    comparison = optimizer.get_business_comparison()
    comparison['cost_matrix'] = {
        'fn_cost': cost_matrix.fn_cost,
        'fp_cost': cost_matrix.fp_cost,
        'asymmetry_ratio': cost_matrix.asymmetry_ratio,
    }

    results_path = RESULTS_DIR / 'threshold_optimization.json'
    with open(results_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\n  [OK] Results saved: {results_path}")

    print("\n  Done.\n")
    return optimizer


if __name__ == '__main__':
    main()
