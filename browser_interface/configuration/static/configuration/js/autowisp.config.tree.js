function startEditNodeText(event) 
{
    let $this = $(this);
    let nodeType = $this.find('.content').text();
    let nodeId = theTree.getEventNode(event).id;
    $('#edit-node').val($this.find('.title').text());
    document.getElementById('edit-node').disabled = 
        !theTree.canEdit
        ||
        (nodeType != 'value' &&  nodeType != 'condition');
    document.getElementById("node-type").innerHTML = 
        nodeType + ":";
    theTree.displayHelp(nodeId);
}


class configTree {
    constructor(data, canEdit)
    {
        this.setIdsAndFlags('', data);
        this.data = data;
        this.createDiagram(false);
        this.canEdit = canEdit

        this.treeDiagram.$chartContainer.on('click', 
                                            '.node',
                                            startEditNodeText);

        if( canEdit ) {
            this.treeDiagram.$chartContainer.on(
                'click', 
                '.bottomEdge',
                function(event) {this.addCondition(event, '>');}.bind(this)
            );
            this.treeDiagram.$chartContainer.on(
                'click', 
                '.topEdge',
                function(event) {this.addCondition(event, '<');}.bind(this)
            );
            this.treeDiagram.$chartContainer.on(
                 'click', 
                 '.rightEdge',
                 function(event) {this.splitCondition(event, 'v');}.bind(this)
            );
            this.treeDiagram.$chartContainer.on(
                 'click', 
                 '.leftEdge',
                 function(event) {this.splitCondition(event, '^');}.bind(this)
            );

            this.treeDiagram.$chartContainer.on('keydown', 
                                                this.handleKeyPress.bind(this));
        }

    }

    alertData()
    {
        alert(JSON.stringify(this.data, null, 4));
    }

    setIdsAndFlags(nodeId, node)
    {
        node.id = nodeId
        if( node.type === 'step' ) {
            node.relationship = '000';
        } else if ( node.type === 'parameter' ) {
            node.relationship = '001'
        } else if ( node.type === 'value' ) {
            node.relationship = '110';
        } else {
            node.relationship = '111';
        } 

        let firstChildId;
        if ( nodeId ) {
            firstChildId = nodeId + '.0'
        } else {
            firstChildId = '0'
        }
        node.children.reduce(this.setIdsAndFlags.bind(this), firstChildId);

        let subIdFrom = nodeId.lastIndexOf(".") + 1
        return nodeId.slice(0, subIdFrom) 
            + 
            (Number(nodeId.slice(subIdFrom)) + 1); 
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

    getParentNode(dataNode)
    {
        return this.getNodeById(
            dataNode.id.slice(0, dataNode.id.lastIndexOf('.'))
        );
    }

    getEventNode(event)
    {
        return this.getNodeById($(event.target).parent()[0].id);
    }

    getIndexInParent(nodeId)
    {
        return Number(nodeId.slice(nodeId.lastIndexOf('.') + 1));
    }

    focusOnNode(nodeId)
    {
        let $chartContainer = $('#chart-container');

        let digger = new JSONDigger(datasource, 
                                    this.$chart.data('options').nodeId, 
                                    'children');
        digger.findNodeById(nodeId).addClass('focused');
    }

    displayHelp(helpParamId)
    {
        if ( helpParamId == undefined ) {
            helpParamId = $('#chart-container').find('.node.focused')[0].id;
        }
        if ( typeof helpParamId == 'string' )
            helpParamId = Number(helpParamId.split('.', 1))
        let paramNode = this.data.children[helpParamId]
        document.getElementById("param-help").innerHTML = 
            "Editing: <h3 style='color:orange'>" + paramNode.name + "</h3>"
            +
            paramNode.description;
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

    splitCondition(event, direction)
    {
        let dataNode = this.getEventNode(event);
        dataNode.relationship = '001';
        let parentNode = this.getParentNode(dataNode);
        let insertPosition = this.getIndexInParent(dataNode.id);
        if ( direction === 'v' ) {
            insertPosition += 1
        }
        parentNode.children.splice(
            insertPosition,
            0,
            {
                'name': 'False',
                'type': 'condition',
                'relationship': '011',
                'children': [
                    {
                        'name': '',
                        'type': 'value',
                        'relationship': '000',
                        'children': []
                    }
                ]
            }
        );
        this.setIdsAndFlags(parentNode.id, parentNode);
        this.createDiagram(true);
    }

    addCondition(event, side)
    {
        let dataNode = this.getEventNode(event);
        let newNode = {
            'name': 'True',
            'type': 'condition'
        }
        if ( side === '<' ) {
            newNode.children = [dataNode];
            this.getParentNode(dataNode).children[
                this.getIndexInParent(dataNode.id)
            ] = newNode;
            this.setIdsAndFlags(dataNode.id, newNode);
        } else {
            newNode.children = dataNode.children
            dataNode.children = [newNode];
            this.setIdsAndFlags(dataNode.id + '.0', newNode)
        }
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

    deleteHighlightedNode()
    {
        let $node = $('#chart-container').find('.node.focused');
        let dataNode = this.getNodeById($node[0].id);
        if ( dataNode.type != 'condition' ) {
            alert('Only conditions can be deleted!') 
            return;
        }
        let parentNode = this.getParentNode(dataNode);
        if ( 
            dataNode.children[0].type == 'value' 
            &&
            parentNode.children.length > 1
        ) {
            dataNode.children = [];
        }
        let indexInParent = this.getIndexInParent(dataNode.id);
        parentNode.children = parentNode.children.slice(
            0, 
            indexInParent
        ).concat(
             dataNode.children,
             parentNode.children.slice(indexInParent + 1)
        );
        this.setIdsAndFlags(parentNode.id, parentNode);
        this.createDiagram(true);
    }

    handleKeyPress(event)
    {
        if ( event.key == 'Delete' ) {
            this.deleteHighlightedNode();
        }
    }

    async save(saveURL)
    {
        postJson(saveURL, this.data).then(function() {location.reload();});
    }
}
