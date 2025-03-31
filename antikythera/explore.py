import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def explore_data(data_path, result_path):
    """
    load and plot the csv data file from the path.
    """
    df = pd.read_csv(data_path)
    sections = sorted(df['Section ID'].unique())
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # map colors to sections
    colors = plt.cm.Set1(np.linspace(0, 1, len(sections)))
    
    # plot the hole lcations
    for i, section in enumerate(sections):
        section_data = df[df['Section ID'] == section]
        ax.scatter(
            section_data['Mean(X)' ], 
            section_data['Mean(Y)'],
            s=80,  
            color=colors[i],
            edgecolors='black',
            alpha=0.7,
            label=f'Section {section} ({len(section_data)} holes)'
        )

    # mark only at the beginning and end of each section
    for section in sections:
        section_data = df[df['Section ID'] == section]
        
        # if only 1 hole, mark directly
        if len(section_data) == 1:
            row = section_data.iloc[0]
            ax.text(row['Mean(X)'], row['Mean(Y)'] + 1, 
                  f"{int(row['Hole'])}", 
                  ha='center', va='center', fontsize=9,
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        # if more than 1 hole, mark at the beginning and end
        else:
            first_row = section_data.iloc[0]
            last_row = section_data.iloc[-1]

            ax.text(first_row['Mean(X)'], first_row['Mean(Y)'] + 1, 
                  f"{int(first_row['Hole'])}", 
                  ha='center', va='center', fontsize=9,
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.1))

            ax.text(last_row['Mean(X)'], last_row['Mean(Y)'] + 1, 
                  f"{int(last_row['Hole'])}", 
                  ha='center', va='center', fontsize=9,
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.1))
            
    # add section labels
    section_labels = []
    for section in sections:
        section_data = df[df['Section ID'] == section]
        
        if len(section_data) > 1:
            x_center = section_data['Mean(X)'].mean()
            y_center = section_data['Mean(Y)'].mean()
    
            section_label = ax.text(
                x_center, y_center + 3,
                f"S{section}",
                ha='center', va='center',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray')
            )
            section_labels.append(section_label)
        
        elif len(section_data) == 1:
            x_center = section_data['Mean(X)'].iloc[0]
            y_center = section_data['Mean(Y)'].iloc[0]
            
            section_label = ax.text(
                x_center, y_center - 2.2,
                f"S{section}",
                ha='center', va='center',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray')
            )
            section_labels.append(section_label)
    

    ax.set_title('Hole Location Measurements', fontsize=16)
    ax.set_xlabel('X (mm)', fontsize=14)
    ax.set_ylabel('Y (mm)', fontsize=14)
    
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.84, 1), loc='upper left')

    plt.tight_layout()
    
    plt.savefig(result_path/'hole_positions.png', dpi=300, bbox_inches='tight')
    
    plt.show()
