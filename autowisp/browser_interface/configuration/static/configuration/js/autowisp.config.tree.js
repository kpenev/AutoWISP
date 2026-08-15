// What the tree shows in place of a value that was deliberately left unset.
const UNDEFINED_VALUE_LABEL = 'undefined (step default)';
const UNDEFINED_VALUE_HINT =
    'undefined: the processing step supplies its own default';


// Is this node a value that was deliberately left undefined?
function isUndefinedValue(dataNode)
{
    return dataNode.type === 'value' && dataNode.name === null;
}


// Only offer to undefine a value that is editable and currently defined.
function showUndefineButton(show)
{
    let button = document.getElementById('undefine-node');
    // Disabled as well as hidden, so the button is still inert if the
    // stylesheet fails to load or is stale in the browser's cache.
    button.disabled = !show;
    button.classList.toggle('hidden-button', !show);
}


function startEditNodeText(event)
{
    let $this = $(this);
    let dataNode = theTree.getNodeById($this[0].id);
    let nodeType = dataNode.type;
    let undefinedValue = isUndefinedValue(dataNode);
    let editField = document.getElementById('edit-node');

    editField.value = undefinedValue ? '' : dataNode.name;
    editField.placeholder = undefinedValue ? UNDEFINED_VALUE_HINT : '';
    editField.disabled =
        !theTree.canEdit
        ||
        (nodeType != 'value' &&  nodeType != 'condition');
    showUndefineButton(
        theTree.canEdit && nodeType == 'value' && !undefinedValue
    );
    document.getElementById("node-type").innerHTML =
        nodeType + ":";
    theTree.displayHelp(dataNode.id);
}


class configTree {
    constructor(data, canEdit)
    {
        this.setIdsAndFlags('', data);
        this.data = data;
        this.createDiagram(false);
        this.canEdit = canEdit

        var chartContainer = document.getElementById('chart-container');
        var savedScroll = sessionStorage.getItem('config_tree_scroll');
        if (savedScroll !== null) {
            sessionStorage.removeItem('config_tree_scroll');
            var pos = JSON.parse(savedScroll);
            chartContainer.scrollLeft = pos.left;
            chartContainer.scrollTop = pos.top;
        }

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
        // The root node's id is the empty string, which the reduce below
        // would read as an index into its own children.
        if ( nodeId === '' )
            return this.data;
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
                'direction': 'l2r',
                'createNode': function($nodeDiv, dataNode) {
                    if ( isUndefinedValue(dataNode) )
                        $nodeDiv.addClass('undefined-value')
                            .children('.title').text(UNDEFINED_VALUE_LABEL);
                }
        });

        // (Re-)building the diagram leaves no node focused, hence no value
        // for the button to act on.
        showUndefineButton(false);
    }

    /** Show the given value node as either undefined or holding its value. */
    styleValueNode($node, dataNode)
    {
        let undefinedValue = isUndefinedValue(dataNode);
        $node.toggleClass('undefined-value', undefinedValue);
        $node.children('.title').text(
            undefinedValue ? UNDEFINED_VALUE_LABEL : dataNode.name
        );
    }

    /** Leave the highlighted value unset, so the step applies its default. */
    setNodeUndefined()
    {
        // A locked configuration must not be modified even if something
        // went wrong with hiding the button.
        if ( !this.canEdit )
            return;
        let $node = $('#chart-container').find('.node.focused');
        let dataNode = this.getNodeById($node[0].id);
        if ( dataNode.type != 'value' ) {
            alert('Only values can be left undefined!');
            return;
        }
        dataNode.name = null;
        this.styleValueNode($node, dataNode);

        let editField = document.getElementById('edit-node');
        editField.value = '';
        editField.placeholder = UNDEFINED_VALUE_HINT;
        showUndefineButton(false);
        hasUnsavedChanges = true;
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
        // Typing a value is how an undefined value gets defined again.
        this.styleValueNode($node, dataNode);
        document.getElementById('edit-node').placeholder = '';
        if ( dataNode.type == 'value' )
            showUndefineButton(this.canEdit);
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
        hasUnsavedChanges = false;
        postJson(saveURL, this.data).then(function() {location.reload();});
    }

    async export()
    {
        const blob = new Blob(
            [JSON.stringify(this.data, null, 4)], 
            {type: "application/json",}
        );
        const fileURL = URL.createObjectURL(blob);
        const downloadLink = document.createElement('a');
        downloadLink.href = fileURL;
        downloadLink.download = 'wisp_config_' + this.data.name + '.json';
        document.body.appendChild(downloadLink);
        downloadLink.click();
        URL.revokeObjectURL(fileURL);
    }
}

// track unsaved changes
let hasUnsavedChanges = false;

// mark form as changed 
function trackFormChanges() 
{
    document.querySelectorAll('input, textarea, select').forEach((el) => {
        el.addEventListener('input', () => { hasUnsavedChanges = true; });
        el.addEventListener('change', () => { hasUnsavedChanges = true; });
});
}

// reset when save configuration is clicked
function registerSaveButtonHandler() 
{
    const saveBtn = document.querySelector('button, input[type="submit"]');
    if (saveBtn) 
    {
        saveBtn.addEventListener('click', () => 
        {
            hasUnsavedChanges = false;
        });
    }
}

// show confirm when trying to leave the page
function handleLinkNavigation(e) 
{
    if (hasUnsavedChanges && !confirm("You have unsaved changes. Are you sure you want to leave?")) 
    {
        e.preventDefault();
    }
}

// show confirm when doing anything else 
function handleBeforeUnload(e) 
{
    if (!hasUnsavedChanges) return;
    const message = 'You have unsaved changes. Are you sure you want to leave?';
    e.preventDefault();
    e.returnValue = message;
    return message;
}


function initUnsavedChangesWarning() 
{
    trackFormChanges();
    registerSaveButtonHandler();

    document.querySelectorAll('a').forEach((link) =>
    {
        link.addEventListener('click', handleLinkNavigation);
    });

    var lockLink = document.getElementById('lock-icon-link');
    if (lockLink) {
        lockLink.addEventListener('click', function () {
            var chartContainer = document.getElementById('chart-container');
            sessionStorage.setItem('config_tree_scroll', JSON.stringify({
                left: chartContainer.scrollLeft,
                top: chartContainer.scrollTop
            }));
        });
    }

    window.addEventListener('beforeunload', handleBeforeUnload);
}


window.addEventListener('DOMContentLoaded', initUnsavedChangesWarning);
