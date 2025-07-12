import subprocess

command = "python -m train_promo --run_name length48_dirichlet_flow_matching_distilled --batch_size 256 --num_workers 4 --num_integration_steps 100 --ckpt workdir/promo_distill_diri_2024-01-09_16-53-39/epoch=14-step=10380.ckpt --validate --validate_on_test --mode distill --sequence_length 48"

subprocess.run(command.split(" "))