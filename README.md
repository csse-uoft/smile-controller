# smile-controller
SMILE Controller



# Setup

#### 1 GraphDB Installation
- [GraphDB](https://www.ontotext.com/products/graphdb/) can be installed for your distribution.
- Make sure it's running port `7200`, e.g. [http://localhost:7200](http://localhost:7200).
- Make sure you have GraphDB running on [http://localhost:7200](http://localhost:7200).

#### 2 GraphDB Repository Creation
- For testing, make sure the username and password are set to `admin`
- Create a new test repository. Go to [http://localhost:7200](http://localhost:7200)
  - Create new repository:
    - Name the repository (Repository ID) as `smile`
    - Set context index to True *(checked).
    - Set query timeout to 45 second.
    - Set the `Throw exception on query timeout` checkmark to True (checked)
    - Click on create repository.
  - Make sure the repository is in "Running" state.


#### 3 GraphDB Configuration
A few notes on configurting SMILE to connect to the database.
- The main SMILE Flask application can be configured in the [src/config/local_config.yml](src/config/local_config.yml) file.
- The config must match ththe knowledge source config.yml file. See the following for testing: [src/config/local_config_test.yml](src/config/local_config_test.yml).

#### 4 Setup smile-controller conda environment
`conda env create -f PySMILEController.yml`

## To run example
1. You will need to start knowledge source listeners.
1. `cd src`
1. `python -i -m main`
1. To view the results, run `show_trace(trace_id)`, where trace_id is the id of the last trace.

Results can be seen in the `<logs>/<trace_id>.txt file`, where `<logs>/` is a folder st up i 
