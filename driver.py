import click
import sys
import train as train_file
# from train import train_model
import subprocess
import pkg_resources
import evaluate_test as eval_test

def train_func(num_epochs):
    """Function to perform training."""
    
    # calling train here

    # num_training, num_epochs, batch_size, learning_rate, weight_decay = train
    # click.echo(f"Training mode is now active. Running {num_training} iterations.")
    click.echo(f"Training mode is now active. Running {num_epochs} iterations.")

    # train_file.main(num_training, num_epochs, batch_size, learning_rate, weight_decay)
    train_file.main(num_epochs)


def eval_func(evaluate):
    """Function to perform evaluation."""

    # calling evaluate here

    count, is_independent = evaluate
    click.echo(f"Evaluating mode is now active. Evaluating {count} instances.")

    eval_test.main(count, is_independent)


def check_dependencies():
    # Attempt to load the requirements.txt file and install each package using pip
    with open("requirements.txt", "r") as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = [pkg for pkg in packages if pkg.split('==')[0].lower() not in installed]

    return missing

    # Setup the use of the model
    # Make GUI and map
    # Output result for how many you want to evaluate

@click.command()
# @click.option('--train', type=(int, int, int, float, float) , help='Activate training mode and numiters/num_epochs/batch_size/learning_rate/weight_decay')
@click.option('--train', type=(int), help='Activate training mode and numiters')
@click.option('--evaluate', type=(int, bool), help='Activate evaluation mode and specify the number of evaluations followed by independence as a boolean.')

def cli(train, evaluate): # , batch_size, learning_rate
    """Simple CLI that can trigger training or evaluation with a specific count."""  
    # checking that only one main option is specified
    if train and evaluate:
        click.echo("Error: Cannot train and evaluate at the same time.", err=True)
        sys.exit(1)
    if not train and not evaluate:
        click.echo("Error: Need to specify either training or evaluate by using --train or --evaluate followed by a number.", err=True)
        sys.exit(1)


    # # this is for checking that all dependencies are present
    # missing = check_dependencies()
    # if missing is not None:
    #     print("Missing packages detected:")
    #     print(missing)
    #     print()

    #     if click.confirm("Do you want to install the missing packages?", default=True):
    #         print("Installing packages...")
    #         python = sys.executable
    #         subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
    #         print("Installation complete.")
    #     else:
    #         print("Installation aborted. The application may not function correctly without the required packages.")
    #         sys.exit(1)

    # this is for if we want to train
    if train is not None:  # Check if --train was provided
        
        # need to fix this
        train_func(train)

    # this is for if we want to evaluate
    if evaluate is not None:  # Check if --evaluate was provided
        eval_func(evaluate)

        # if (batch_size is None and learning_rate is None): 
        #     eval_func(evaluate)
        # elif batch_size is not None: 
        #     eval_func(evaluate, learning_rate)


if __name__ == '__main__':
    cli()
