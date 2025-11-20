from datetime import date
import pandas as pd

index = pd.date_range(start='12/24/2025', end='1/6/2027')

sonntage = index.weekday == 6

erster_tag = index.day == 1

erster_sonntag = index[sonntage & erster_tag]

print(erster_sonntag)
print(f"Vom 24.12.2025 bis zum 6.01.2027 gibt es {len(erster_sonntag)} Sonntage, die auf den 1. des Monats fallen.")


#print(f"Vom {index[0].date()} bis zum {index[-1].date()} gibt es {len(erster_sonntag)} Sonntage, die auf den 1. des Monats fallen.")