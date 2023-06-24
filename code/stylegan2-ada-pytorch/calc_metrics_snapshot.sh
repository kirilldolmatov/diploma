
# calculate metric for each snapshot
for file in /media/kirill/2tb/diploma/experiments/15/*.pkl
do
    python calc_metrics.py --metrics=fid50k_full --network="$file" | tee -a fid_results_exp15.txt
done
