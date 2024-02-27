function startEditNodeText() 
{
    let $this = $(this);
    let nodeType = $this.find('.content').text()
    $('#edit-node').val($this.find('.title').text());
    document.getElementById('edit-node').disabled = 
        (nodeType != 'value' &&  nodeType != 'condition');
    document.getElementById("node-type").innerHTML = 
        nodeType + ":"
}


function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}


class configTree {
    constructor(data, saveURL)
    {
        this.addFlags(data)
        this.data = data
        this.saveURL = saveURL
        this.createDiagram(false);

        this.treeDiagram.$chartContainer.on('click', 
                                            '.node',
                                            startEditNodeText);

        this.treeDiagram.$chartContainer.on('click', 
                                            '.bottomEdge',
                                            this.addCondition.bind(this));
        this.treeDiagram.$chartContainer.on('click', 
                                            '.rightEdge',
                                            this.splitCondition.bind(this));


    }

    alertData()
    {
        alert(JSON.stringify(this.data, null, 4));
    }

    addFlags(node)
    {
        if( node.type === 'step' ) {
            node.relationship = '000';
        } else if ( node.type === 'value' ) {
            node.relationship = '000';
        } else {
            node.relationship = '001';
        } 
        node.children.forEach(this.addFlags.bind(this));
        let num_children = node.children.length
        if ( num_children > 0 ) {
            node.children[num_children - 1].relationship = '011';
        }
    }

    getNodeById(nodeId)
    {
        return nodeId.split('.').reduce(
            function (subTree, childIndexStr) {
                return subTree.children[Number(childIndexStr)];
            },
            this.data
        );
    }

    createDiagram(deleteFirst)
    {

        let $chartContainer = $('#chart-container');
        if ( deleteFirst ) {
            this.treeDiagram.removeNodes($chartContainer.find('.node:first'));
        }

        this.treeDiagram = $chartContainer.orgchart({
                'data' : this.data,
                'nodeContent': 'type',
                'direction': 'l2r'
        });
    }

    splitCondition(event)
    {
        let dataNode = this.getNodeById($(event.target).parent()[0].id);
        dataNode.relationship = '001';
        let parentNode = this.getNodeById(
            dataNode.id.slice(0, dataNode.id.lastIndexOf('.'))
        );
        let newNodeId = parentNode.id + '.' + parentNode.children.length;
        parentNode.children.push(
            {
                'name': 'False',
                'type': 'condition',
                'id': newNodeId,
                'relationship': '011',
                'children': [
                    {
                        'name': '',
                        'type': 'value',
                        'id': newNodeId + '.0',
                        'relationship': '000',
                        'children': []
                    }
                ]
            }
        );
        this.createDiagram(true);
    }

    addCondition(event)
    {
        let dataNode = this.getNodeById($(event.target).parent()[0].id);
        let newNode = {
            'name': 'True',
            'type': 'condition',
            'id': dataNode.id + '.0',
            'relationship': '011',
            'children': dataNode.children
        }
        newNode.children.forEach(function (child) {
            child.id = newNode.id + child.id.slice(dataNode.id.length);
        });
        dataNode.children = [newNode];
        this.createDiagram(true);
    }

    changeNodeText() 
    {
        let $node = $('#chart-container').find('.node.focused');
        let dataNode = this.getNodeById($node[0].id);
        dataNode.name = $('#edit-node').val();
        $node.find('.title').text(
             $('#edit-node').val()
        );
    }

    save()
    {
        let csrftoken = getCookie('csrftoken');
        let headers = new Headers();
        headers.append('X-CSRFToken', csrftoken);
        headers.append("Content-type", "application/json; charset=UTF-8")
        fetch(this.saveURL, {
            method: "POST",
            body: JSON.stringify(this.data),
            headers: headers,
            credentials: 'include'
        })
            .then((response) => response.body)

    }
}
