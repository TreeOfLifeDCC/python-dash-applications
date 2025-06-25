# python-dash-applications
Repository to store python-dash applications

## Running in Docker

```bash
docker build -t python-dash-applications .
docker run -it --rm -p 8000:80 python-dash-applications
```

## Accessing the App
Once the container is running, open your browser and navigate to:
http://localhost:8000

## Accessing Trees
Default tree (DTOL project):<br>
http://localhost:8000/cytoscape

Other project trees:<br>
http://localhost:8000/cytoscape?projectName=dtol
http://localhost:8000/cytoscape?projectName=erga
http://localhost:8000/cytoscape?projectName=asg
http://localhost:8000/cytoscape?projectName=gbdp
