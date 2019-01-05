#!/bin/bash

USERNAME=wopauli@microsoft.com

AZURE_REGION=westus

FILE=token.txt

if [ -f $FILE ]; then
    echo "Token file $FILE exists."
    token=$(cat $FILE)
else
    echo "Token file $FILE does not exist."
    exit 1
fi


echo "Using access token: $token"

# curl -n -o notebooks.dbc   'https://${AZURE_REGION}.azuredatabricks.net/api/2.0/workspace/export?path=/Users/$USERNAME/notebooks&direct_download=true&format=DBC' -H "Authorization: Bearer ${token}"

# Here are some further instructions

# To extract the content of the archive, simply use unzip:
# > unzip notebooks.dbc

# To create a dbc archive of folders, just create a zip file without compressing python notebooks (-n option)
# > zip -r -n python notebooks notebooks

# To import this new archive into databricks, use curl again
# > curl -n -F path=/Users/${USERNAME}/notebooks -F format=DBC -F content=@notebooks.dbc https://${AZURE_REGION}.azuredatabricks.net/api/2.0/workspace/import  -H "Authorization: Bearer ${token}"
