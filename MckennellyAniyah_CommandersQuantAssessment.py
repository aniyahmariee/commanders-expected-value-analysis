#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages


# In[11]:


# In[2]:
url = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.csv.gz"
pbp = pd.read_csv(url, low_memory=False)

# === CRITICAL: REGULAR SEASON ONLY (this is what matches the ground truth) ===
pbp = pbp[pbp['season_type'] == 'REG']
print(f"Loaded regular season only: {len(pbp):,} rows")


# In[12]:


teams = ["WAS", "PHI", "IND"]
team_names = {
    "WAS": "Washington Commanders",
    "PHI": "Philadelphia Eagles",
    "IND": "Indianapolis Colts",
}

THEMES = {
    "WAS": {"hdr_dark":"#5A1414","hdr_mid":"#7A1E1E","hdr_light":"#9B2828","accent":"#FFB612","row_alt":"#FDF6E3"},
    "PHI": {"hdr_dark":"#004C54","hdr_mid":"#006672","hdr_light":"#008080","accent":"#A5ACAF","row_alt":"#EEF5F5"},
    "IND": {"hdr_dark":"#002C5F","hdr_mid":"#003D7A","hdr_light":"#0050A0","accent":"#A2AAAD","row_alt":"#EEF2F8"},
}

df = pbp[
    (pbp["posteam"].isin(teams)) &
    (pbp["play_type"].isin(["run", "pass"])) &
    (pbp["down"].isin([1, 2, 3, 4])) &
    (pbp["yards_gained"].notna()) &
    (pbp["penalty"] == 0)
].copy()

# Exclude red zone (+1 to +20 yards from opponent EZ)
df = df[~((df["yardline_100"] >= 1) & (df["yardline_100"] <= 20))]

# Exclude backed up (90-100 yards from opponent EZ)
df = df[~((df["yardline_100"] >= 90) & (df["yardline_100"] <= 100))]

print(f"Filtered to {len(df):,} plays")


# In[13]:


def ydstogo_bucket(down, ytg):
    ytg = int(ytg)
    if down == 1:
        return "1st & 10" if ytg == 10 else None
    elif down == 2:
        if ytg <= 2:   return "2nd & 1-2"
        elif ytg <= 6: return "2nd & 3-6"
        else:          return "2nd & 7+"
    elif down == 3:
        if ytg <= 2:   return "3rd & 1-2"
        elif ytg <= 5: return "3rd & 3-5"
        elif ytg <= 9: return "3rd & 6-9"
        else:          return "3rd & 10+"
    elif down == 4:
        if ytg <= 2:   return "4th & 1-2"
        else:          return "4th & 3+"
    return None

df["label"] = df.apply(lambda r: ydstogo_bucket(r["down"], r["ydstogo"]), axis=1)
df = df[df["label"].notna()].copy()

ROW_ORDER = [
    "1st & 10",
    "2nd & 1-2", "2nd & 3-6", "2nd & 7+",
    "3rd & 1-2", "3rd & 3-5", "3rd & 6-9", "3rd & 10+",
    "4th & 1-2", "4th & 3+",
]


# In[14]:


def build_summary(team_df):
    rows = []
    for label in ROW_ORDER:
        sub  = team_df[team_df["label"] == label]
        runs = sub[sub["play_type"] == "run"]
        pas  = sub[sub["play_type"] == "pass"]
        n_total, n_run, n_pass = len(sub), len(runs), len(pas)

        rows.append([
            label,
            f"{round(n_run/n_total*100)}%" if n_total > 0 else "0%",
            str(n_run),
            f"{runs['yards_gained'].mean():.1f}" if n_run  > 0 else "-",
            str(n_pass),
            f"{pas['yards_gained'].mean():.1f}"  if n_pass > 0 else "-",
            str(n_total),
            f"{sub['yards_gained'].mean():.1f}"  if n_total > 0 else "-",
        ])

    # Totals row
    t_run  = team_df[team_df["play_type"] == "run"]
    t_pass = team_df[team_df["play_type"] == "pass"]
    t_total = len(team_df)
    rows.append([
        "Total",
        f"{round(len(t_run)/t_total*100)}%" if t_total > 0 else "0%",
        str(len(t_run)),
        f"{t_run['yards_gained'].mean():.1f}"   if len(t_run)  > 0 else "-",
        str(len(t_pass)),
        f"{t_pass['yards_gained'].mean():.1f}"  if len(t_pass) > 0 else "-",
        str(t_total),
        f"{team_df['yards_gained'].mean():.1f}" if t_total > 0 else "-",
    ])
    return rows


# In[15]:


def draw_table(team_abbr, team_name, team_df):
    rows = build_summary(team_df)
    theme = THEMES[team_abbr]
    HDR_DARK  = theme["hdr_dark"]
    HDR_MID   = theme["hdr_mid"]
    HDR_LIGHT = theme["hdr_light"]
    ACCENT    = theme["accent"]
    ROW_ALT   = theme["row_alt"]
    ROW_WHITE = "#ffffff"
    BORDER    = "#cccccc"
    TXT_WHITE = "#ffffff"
    TXT_DARK  = "#1a1a1a"

    n_data = len(ROW_ORDER)
    col_labels = ["Down &\nDistance","Run %","Plays","Yards /\nPlay","Plays","Yards /\nPlay","Plays","Yards /\nPlay"]
    col_widths = [0.155,0.075,0.085,0.115,0.085,0.115,0.085,0.115]
    col_widths[-1] += 1.0 - sum(col_widths)
    n_cols = len(col_labels)
    ROW_H=0.055; TOP_H=0.10; FOOT_H=0.05
    fig_h=(TOP_H+(n_data+2)*ROW_H+FOOT_H+0.02)*10

    fig = plt.figure(figsize=(10, fig_h))
    ax  = fig.add_axes([0.03, 0.02, 0.94, 0.96])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

    def col_left(c): return sum(col_widths[:c])
    def row_top(r):  return 1 - TOP_H - r * ROW_H

    def fill_rect(c0, c1, r_top, r_bot, color):
        x=col_left(c0); w=col_left(c1)-x
        ax.add_patch(mpatches.Rectangle((x,r_bot),w,r_top-r_bot,
            facecolor=color,edgecolor="none",zorder=1,transform=ax.transAxes))

    def cell_text(text,c0,c1,r_top,r_bot,color=TXT_DARK,fontsize=8,bold=False,ha="center"):
        x=(col_left(c0)+col_left(c1))/2; y=(r_top+r_bot)/2
        ax.text(x,y,text,color=color,fontsize=fontsize,
                fontweight="bold" if bold else "normal",
                ha=ha,va="center",transform=ax.transAxes,zorder=2)

    # Title block
    fill_rect(0,n_cols,1,1-TOP_H,HDR_DARK)
    ax.add_patch(mpatches.Rectangle((0.05,1-TOP_H*0.52),0.90,0.003,
        facecolor=ACCENT,edgecolor="none",zorder=3,transform=ax.transAxes))
    ax.text(0.5,1-TOP_H*0.28,"Offensive Run / Pass Summary",
            color=TXT_WHITE,fontsize=13,fontweight="bold",
            ha="center",va="center",transform=ax.transAxes,zorder=2)
    ax.text(0.5,1-TOP_H*0.68,f"{team_name} – 2025 Regular Season",
            color=ACCENT,fontsize=9,fontweight="bold",
            ha="center",va="center",transform=ax.transAxes,zorder=2)

    # Group header row
    r0,r1=row_top(0),row_top(1)
    fill_rect(0,2,r0,r1,HDR_DARK); fill_rect(2,4,r0,r1,HDR_MID)
    fill_rect(4,6,r0,r1,HDR_MID);  fill_rect(6,8,r0,r1,HDR_MID)
    cell_text("Run Plays", 2,4,r0,r1,TXT_WHITE,bold=True,fontsize=9)
    cell_text("Pass Plays",4,6,r0,r1,TXT_WHITE,bold=True,fontsize=9)
    cell_text("All Plays", 6,8,r0,r1,TXT_WHITE,bold=True,fontsize=9)

    # Sub-header row
    r0,r1=row_top(1),row_top(2)
    fill_rect(0,n_cols,r0,r1,HDR_LIGHT)
    for c,lbl in enumerate(col_labels):
        cell_text(lbl,c,c+1,r0,r1,TXT_WHITE,bold=True,fontsize=7.5)

    # Data rows
    for i,row_data in enumerate(rows[:-1]):
        r0,r1=row_top(i+2),row_top(i+3)
        fill_rect(0,n_cols,r0,r1,ROW_ALT if i%2==1 else ROW_WHITE)
        for c,val in enumerate(row_data):
            if c==0:
                ax.text(col_left(0)+0.008,(r0+r1)/2,val,color=HDR_DARK,fontsize=8,
                        fontweight="bold",ha="left",va="center",transform=ax.transAxes,zorder=2)
            elif c==1:
                cell_text(val,c,c+1,r0,r1,HDR_DARK,bold=True,fontsize=8)
            else:
                cell_text(val,c,c+1,r0,r1,TXT_DARK,fontsize=8)

    # Total row
    r0,r1=row_top(n_data+2),row_top(n_data+3)
    fill_rect(0,n_cols,r0,r1,HDR_DARK)
    ax.add_patch(mpatches.Rectangle((0,r1),0.008,r0-r1,
        facecolor=ACCENT,edgecolor="none",zorder=3,transform=ax.transAxes))
    for c,val in enumerate(rows[-1]):
        cell_text(val,c,c+1,r0,r1,TXT_WHITE,bold=True,fontsize=8.5)

    # Grid lines
    for i in range(2,n_data+3):
        y=row_top(i)
        ax.plot([0,1],[y,y],color=BORDER,lw=0.5,transform=ax.transAxes,zorder=3)
    for c in [2,4,6]:
        x=col_left(c)
        ax.plot([x,x],[row_top(n_data+3),row_top(0)],
                color=HDR_DARK,lw=1.5,transform=ax.transAxes,zorder=3)

    # Outer border
    y_bot=row_top(n_data+3); y_top=row_top(0)
    ax.add_patch(mpatches.Rectangle((0,y_bot),1,y_top-y_bot,
        facecolor="none",edgecolor=HDR_DARK,lw=2,transform=ax.transAxes,zorder=4))

    # Footnote
    ax.text(0.0,row_top(n_data+3)-0.012,
            "Excludes red zone (+1 to +20), backed up (-1 to -10), "
            "plays with penalties, and non-standard down and distances.",
            color="#666666",fontsize=6.5,style="italic",
            ha="left",va="top",transform=ax.transAxes)

    fig.patch.set_facecolor("white")
    return fig


# In[17]:


pdf_path = "MckennellyAniyah_CommandersQuantAssessment.pdf"

with PdfPages(pdf_path) as pdf:
    for abbr, name in team_names.items():
        team_df = df[df["posteam"] == abbr]
        fig = draw_table(abbr, name, team_df)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓  {name}")

print(f"\nSaved → {pdf_path}")

