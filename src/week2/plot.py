import argparse, os, pandas as pd, matplotlib.pyplot as plt

def load(csvs):
    dfs = []
    for p in csvs:
        if os.path.exists(p): dfs.append(pd.read_csv(p))
    if not dfs: raise FileNotFoundError('No CSVs found.')
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='comma-separated CSVs')
    ap.add_argument('--out_dir', default='reports/figs_week2')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load([x.strip() for x in args.csv.split(',')])

    plt.figure()
    for m in sorted(df['method'].unique()):
        sub = df[df['method']==m]
        plt.scatter(sub['tuned_params_%'], sub['acc@1'], label=m)
    plt.xlabel('% Params Tuned'); plt.ylabel('Accuracy@1')
    plt.title('Accuracy vs Tuned Params (%)')
    plt.grid(True, linestyle=':'); plt.legend()
    plt.savefig(os.path.join(args.out_dir, 'acc_vs_params.png'), bbox_inches='tight'); plt.close()

    plt.figure()
    for m in sorted(df['method'].unique()):
        sub = df[df['method']==m]
        plt.scatter(sub['acc@1'], sub['ece'], label=m)
    plt.xlabel('Accuracy@1'); plt.ylabel('ECE')
    plt.title('ECE vs Accuracy')
    plt.grid(True, linestyle=':'); plt.legend()
    plt.savefig(os.path.join(args.out_dir, 'ece_vs_acc.png'), bbox_inches='tight'); plt.close()

    print(f"Saved to {args.out_dir}")

if __name__ == '__main__':
    main()
