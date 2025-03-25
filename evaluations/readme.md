First change the data path in the code, then perform evaluation.

## clip_score
```
cd clip_scores
CUDA_VISIBLE_DEVICES=0 python compute_clip_scores.py --method ours
```

## FID/KID
```
cd fid_scores
CUDA_VISIBLE_DEVICES=0 bash fid.sh
CUDA_VISIBLE_DEVICES=0 bash kid.sh
```


