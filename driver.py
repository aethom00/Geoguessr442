import click
import sys
from train import train_model
import subprocess
import pkg_resources

def train_func(count, lr = 1e-3):
    """Function to perform training."""
    click.echo(f"Training mode is now active. Running {count} iterations.")
    
    # calling train here
    train_model()


def eval_func(count):
    """Function to perform evaluation."""
    click.echo(f"Evaluating mode is now active. Evaluating {count} instances.")

    # calling evaluate here
        


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
@click.option('--train', type=int, help='Activate training mode and specify the number of iterations.')
@click.option('--evaluate', type=int, help='Activate evaluation mode and specify the number of evaluations.')
# @click.option('--batch-size', type=int, default=32, help='Specify batch size for training.')
@click.option('--learning-rate', type=float, default=0.001, help='Specify learning rate for training.')

def cli(train, evaluate, batch_size, learning_rate):
    """Simple CLI that can trigger training or evaluation with a specific count."""
    if train: 
        # if batch_size:
        #     if batch_size.type != int:
        #         click.echo("Error: batch size must be an integer.", err=True)
        #         sys.exit(1)
        if learning_rate:
            if learning_rate.type != float: 
                click.echo("Error: learning rate must be a float.", err=True)
                sys.exit(1)

    if evaluate: 
        if batch_size or learning_rate: 
            click.echo("Error: Cannot evaluate with batch size or learning raye.", err=True)
            sys.exit(1)      






    # checking that only one main option is specified
    if train and evaluate:
        click.echo("Error: Cannot train and evaluate at the same time.", err=True)
        sys.exit(1)
    if not train and not evaluate:
        click.echo("Error: Need to specify either training or evaluate by using --train or --evaluate followed by a number.", err=True)
        sys.exit(1)


    # this is for checking that all dependencies are present
    missing = check_dependencies()
    if missing is not None:
        print("Missing packages detected:")
        print(missing)
        print()

        if click.confirm("Do you want to install the missing packages?", default=True):
            print("Installing packages...")
            python = sys.executable
            subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
            print("Installation complete.")
        else:
            print("Installation aborted. The application may not function correctly without the required packages.")
            sys.exit(1)

    # this is for if we want to train
    if train is not None:  # Check if --train was provided
        train_func(train)

    # this is for if we want to evaluate
    if evaluate is not None:  # Check if --evaluate was provided
        if (batch_size is None and learning_rate is None): 
            eval_func(evaluate)
        elif batch_size is not None: 
            eval_func(evaluate, learning_rate)


if __name__ == '__main__':
    cli()
