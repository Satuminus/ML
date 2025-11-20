from datetime import date
import pandas as pd

index = pd.date_range(start='12/24/2025', end='1/6/2026', freq='B')
print(index)
print(f"Es gibt {len(index)} normale Wochentage/reguläre Arbeitstage in den Weihnachtsferien 2025/26.")