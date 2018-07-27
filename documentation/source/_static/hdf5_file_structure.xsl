<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:output method="html" encoding="utf-8" indent="yes"/>


<xsl:variable name="itemid" select="0" /> 

<xsl:template match="/group">
    <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1"/>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"/>
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"/>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"/>

            <style>
                body {
                    color: lightgrey;
                    background: black;
                }
                dd {
                    padding-left: 3em
                }
                code {
                    background: black;
                }
            </style>

        </head>
        <body bgcolor="black">
            <h1 style="color:white">
                <xsl:value-of select="@name"/>
                (v. <xsl:value-of select="@version"/>)
            </h1>
            <hr/>
            <dl>
                <xsl:apply-templates/>
            </dl>
        </body>
    </html>
</xsl:template>

<xsl:template match="dataset">
    <xsl:param name="parent"/>
    <xsl:variable name="uniqueid">
        <xsl:value-of select="$parent"/>
        <xsl:number value="position()" format="1" />
    </xsl:variable>

    <dt>
        <a data-toggle="collapse">
            <xsl:attribute name="href">
                #<xsl:value-of select="$uniqueid"/>
            </xsl:attribute>
            <span style="color:lime;font-size:150%;line-height:150%">
                <strong><xsl:value-of select="@name"/></strong>
                (<code><xsl:value-of select="@key"/></code>)
            </span>
            <code>
                [<xsl:value-of select="@dtype"/>]
                <xsl:if test="@compression != ':'">
                    [<xsl:value-of select="@compression"/>]
                </xsl:if>
                <xsl:if test="@scaleoffset != 'None'">
                    [scaleoffset:<xsl:value-of select="@scaleoffset"/>]
                </xsl:if>
                <xsl:if test="@shuffle != 'False'">
                    [shuffle]
                </xsl:if>
                <xsl:if test="@fill != 'None'">
                    [fill:<xsl:value-of select="@fill"/>]
                </xsl:if>
            </code>
        </a>
    </dt>
    <dd class="collapse in">
        <xsl:attribute name="id">
            <xsl:value-of select="$uniqueid"/>
        </xsl:attribute>
        <xsl:value-of select="@description"/>
        <dl>
            <xsl:apply-templates/>
        </dl>
    </dd>
</xsl:template>

<xsl:template match="attribute">
    <dt>
        <span style="color:gold">
            <strong><xsl:value-of select="@name"/></strong>
            (<code style="font-size:150%"><xsl:value-of select="@key"/></code>)
        </span>
        <code>
            [<xsl:value-of select="@dtype"/>]
        </code>
    </dt>
    <dd>
        <xsl:value-of select="@description"/>
    </dd>
</xsl:template>

<xsl:template match="link">
    <dt>
        <span style="color:cyan">
            <xsl:value-of select="@name"/>
            (<code style="font-size:150%"><xsl:value-of select="@key"/></code>)
        </span>
        -->
        <mark>
            <code style="font-size:150%"><xsl:value-of select="@target"/></code>
        </mark>
    </dt>
    <dd>
        <xsl:value-of select="@description"/>
    </dd>
</xsl:template>

<xsl:template match="group">
    <xsl:param name="parent"/>
    <xsl:variable name="uniqueid">
        <xsl:value-of select="$parent"/>
        <xsl:number value="position()" format="1" />
    </xsl:variable>

    <dt>
        <a data-toggle="collapse">
            <xsl:attribute name="href">
                #<xsl:value-of select="$uniqueid"/>
            </xsl:attribute>
            <span style="color:dodgerblue;font-size:200%;line-height:150%">
                <xsl:value-of select="@name"/>
            </span>
        </a>
    </dt>
    <dd class="collapse in">
        <xsl:attribute name="id">
            <xsl:value-of select="$uniqueid"/>
        </xsl:attribute>
        <dl>
            <xsl:apply-templates>
                <xsl:with-param name="parent">
                    <xsl:value-of select="$uniqueid"/>
                </xsl:with-param>
            </xsl:apply-templates>
        </dl>

    </dd>
</xsl:template>

</xsl:stylesheet>
