# Notebook runner
# Use source to run

function run_notebook
  set -l script (mktemp -u -p ./)
  jupyter nbconvert --stdout --to script $argv[1] > $script

  sed -i "/get_ipython()/d" $script

  # Assuming region is set to nat
  sed -i "s/REGION = \"nat\"/REGION = \"$argv[2]\"/" $script

  # Assuming target is set to initial value
  sed -i "s/TARGET = \"1-ahead\"/TARGET = \"$argv[3]\"/" $script

  # Run stuff
  python $script

  # Start new
  rm -f $script
end

for r in "nat" "hhs1" "hhs2" "hhs3" "hhs4" "hhs5" "hhs6" "hhs7" "hhs8" "hhs9" "hhs10"
    for t in "1-ahead" "2-ahead" "3-ahead" "4-ahead"
        echo "Running for target $t and region $r"
        run_notebook ./04-train-multibin.ipynb $r $t
    end
end
