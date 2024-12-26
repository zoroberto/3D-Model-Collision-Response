  using System.Collections;
using System.Collections.Generic;
using ExporterImporter;
using Octree;
using PBD;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Serialization;

public class GPURigid : MonoBehaviour
{
    private static readonly int CollisionBox = Shader.PropertyToID("collisionBox");

    public enum MyModel
    {
        Cube,
        IcoSphere_low,
        Torus,
        Bunny,
        Armadillo,
        Dragon_Vrip,
        Dragon_Refine,
        Horse,
        Asian_Dragon,
        Lucy,
    };

    public enum OctLevel
    {
        // None,
        Zero,
        One,
        Two
    };

    [Header("3D model")]
    public MyModel model;
   
    [HideInInspector]
    private string modelName;

    [Header("Obj Parameters")]
    [SerializeField] int numberOfObjects = 1;
    [SerializeField] float invMass = 1.0f;
    [SerializeField] Vector3 gravity = new Vector3(0.0f, -9.81f, 0.0f);
    [SerializeField] float dt = 0.01f; 
    [SerializeField] float collisionDamping = 0.8f; // velocity damping
    
    [Header("Collision Response")]
    [SerializeField] float separationBias = 0.001f;
    [SerializeField] float dampingPosition = 0.01f; 
    [SerializeField] float restitution = 0.2f; 
    [SerializeField] float dampingVelocity = 0.8f; 
    [SerializeField] float impulseFactor = 0.2f; 

    [Header("Import CSV")]
    public string csv_file = "object_positions.csv";

    [Header("Collision Boundary")]
    public GameObject boundary;

    [Header("Rendering Paramenters")]
    public ComputeShader computeShader;
    public Shader renderingShader;
    public Color matColor;
    
    [HideInInspector]
    private Material material;
    private ComputeShader computeShaderObj;

    [HideInInspector]
    private int nodeCount;
    private int triCount; // size of triangle
    
    //for render
    private ComputeBuffer vertsBuff;
    private ComputeBuffer triBuffer;
    //for compute shader
    private ComputeBuffer positionsBuffer;
    private ComputeBuffer velocitiesBuffer ;
    private ComputeBuffer triangleBuffer;
    private ComputeBuffer triangleIndicesBuffer;
    
    private ComputeBuffer boundaryBBBuffer;
    private ComputeBuffer boundaryPositionsBuffer;
    private ComputeBuffer bbMinMaxBuffer;
    private ComputeBuffer objectIndexBuffer;
    private ComputeBuffer bbOctreeBuffer;
    private ComputeBuffer randomVelocitiesBuffer;
    
    private ComputeBuffer octreeIndicesBuffer;
    private ComputeBuffer relationVertexAABBBuffer;
    
    private ComputeBuffer collisionBoxBuffer;
    private ComputeBuffer collisionNodeBuffer;
    private ComputeBuffer collisionTestingBuffer;
    private ComputeBuffer nodeInsideBoxBuffer; // for rendering purpose
    private ComputeBuffer collidedNodeFlagBuffer;
    
    // for counter interlocked function
    private ComputeBuffer nodeIndexBuffer;
    private ComputeBuffer nodeCounterBuffer;
    private ComputeBuffer nodeProcessedUniqueBuffer;
    
    private ComputeBuffer allNodeCounterBuffer;
    private ComputeBuffer allNodeIndexBuffer;
    private ComputeBuffer nodeUniqueBuffer;
    
    
    // executing kernels
    private int computeVerticesNormal; // for rendering purpose 
    private int updatePosKernel;
    private int CollisionBoundaryHandlingKernel;
    private int findBoundaryMinMaxKernel;
    private int findBBMinMaxKernel; 
    private int ImplementOctreeKernel;
    private int UpdateVertexAndBoxRelationKernel;
    private int CheckBBCollisionKernel;
    private int rmCollisionsBoxKernel;
    private int rmNodeCollisionKernel;
    private int rmNodeCounterBufferKernel;
    private int UpdateNodeInsideBoxKernel;
    private int CollisionResponseKernel;
    private int UpdateCollidedNodeFlagKernel;
    private int CollisionDetectionResponseKernel;
    

    private List<Triangle> triangles = new List<Triangle>();
    private Vector3[] Positions;
    private Vector3[] Velocities;
    private Vector2Int[] indicies;
    
    private struct vertData
    {
        public Vector3 pos;
        public Vector2 uvs;
        public Vector3 norms;
    };
    private int[] triArray;
    private vertData[] vDataArray;
    

    // octree
    private List<int> octreeIndices = new List<int>();
    private readonly int octree_size = 73;
    
    // fetch data array
    private OctreeData[] bbOctree;
    private int[] relationVertexAABB;
    private int[] collisionBox;
    private int[] nodeInsideBox;
    private int[] nodeCounterData;
    private int[] nodeIndex;
    private int[] allNodeIndex;
    private int[] counterData;
    private Vector3[] nodePositions;
    private Vector2Int[] collisionNode;
    private List<int> triIntersectionCounting = new List<int>();
    private int[] collisionTesting;
    private int[] indexData;

    [HideInInspector] // randomly movement
    private Vector3[] movementDirection;
    private BoundData[] boundaryMinMax;
    private Vector3[] randomVelocity;


    [Header("Debug Mode")]
    public OctLevel octreeLevel;
    public bool wholeNodes;
    public bool debugNode; 
    public bool debugBox;
    private bool boxLv0; 
    private bool boxLv1; 
    private bool boxLv2; 

    private float deltaTime;
    
    
    void Start()
    {
        material = new Material(renderingShader); // new material for difference object
        material.color = matColor; //set color to material
        computeShaderObj = Instantiate(computeShader); // to instantiate the compute shader to be use with multiple object

        SelectModelName();
        SelectOctreeLevel();
        AddOctreeIndices();
        FindBoundaryMinMax();
        SetupMeshData();
        SetupShader();
        SetBuffData();
        SetupMeshComputeBuffer();
        SetupCollisionComputeBuffer();
        SetupKernel();
        StartCoroutine(SetupComputeShader());
    }

    void SelectModelName()
    {
        switch (model)
        {
            case MyModel.Cube: modelName = "33cube.1"; break;
            case MyModel.IcoSphere_low: modelName = "icosphere_low.1"; break;
            case MyModel.Torus: modelName = "torus.1"; break; // size 0.2 on the scene
            case MyModel.Bunny: modelName = "bunny.1"; break;
            case MyModel.Armadillo: modelName = "Armadillo.1"; break; // size 0.15 on the scene
            case MyModel.Dragon_Vrip: modelName = "stanford-dragon-vrip-res-3.1"; break; // size 3 on the scene
            case MyModel.Dragon_Refine: modelName = "dragon_refine_01.1"; break;
            case MyModel.Horse: modelName = "horse.1"; break; // size 3 on the scene
            case MyModel.Asian_Dragon: modelName = "asian_dragon.1"; break; // size 0.003 on the scene
            case MyModel.Lucy: modelName = "lucy.1"; break; // size 0.005 on the scene
        }
    }
    
    private void SelectOctreeLevel() 
    {
        switch(octreeLevel)   
        {
            // case OctLevel.None : break;
            case OctLevel.Zero : boxLv0 = true; boxLv1 = false; boxLv2 = false; break;
            case OctLevel.One  : boxLv1 = true; boxLv0 = false; boxLv2 = false; break;
            case OctLevel.Two  : boxLv2 = true; boxLv0 = false; boxLv1 = false; break;
        }
    }

    
    int calculateStartIndex(int object_index, int level)
    {
        if (level == 1) return object_index * octree_size + 1;
        if (level == 2) return object_index * octree_size + 9;
        return object_index * octree_size ;
    }

    int calculateEndIndex(int object_index, int level)
    {
        if (level == 1) return object_index * octree_size + 9;
        if (level == 2) return object_index * octree_size + 73;
        return object_index * octree_size ;
    }

    private void AddOctreeIndices()
    {
        for (int i = 0; i < numberOfObjects; i++)
        {
            var l1_st_idx = calculateStartIndex(i, 1);
            var l1_end_idx = calculateEndIndex(i, 1);
            var l2_st_idx = calculateStartIndex(i, 2);
            var l2_end_idx = calculateEndIndex(i, 2);

            if(boxLv0)
            {
                octreeIndices.Add(i * octree_size);

                
            }
            
            if(boxLv1)
            {
                for(int s = l1_st_idx; s < l1_end_idx; s++) 
                {
                    octreeIndices.Add(s);
                }
            }
            
            if(boxLv2)
            {
                for(int s = l2_st_idx; s < l2_end_idx; s++) 
                {
                    octreeIndices.Add(s);
                }
            }

            if(!(boxLv0 || boxLv1 || boxLv2))
            {
                 octreeIndices.Add(-1);
            }
        }
        
        for (int j = 0; j < octreeIndices.Count; j++)
        {
            // print($"octree index {j} {octreeIndices[j]}");
        }
    }

    private void FindBoundaryMinMax()
    {
        Vector3[] _boundaryVertices = boundary.GetComponent<MeshFilter>().mesh.vertices;
        List<Vector3> boundaryVertices = new List<Vector3>();

        for (int i = 0; i < _boundaryVertices.Length; i++)
        {
            boundaryVertices.Add(boundary.transform.TransformPoint(_boundaryVertices[i]));
        }
        
        boundaryMinMax = new BoundData[1];
        boundaryBBBuffer = new ComputeBuffer(1, sizeof(float) * 6);
        boundaryPositionsBuffer = new ComputeBuffer(_boundaryVertices.Length, sizeof(float) * 3);
        boundaryPositionsBuffer.SetData(boundaryVertices);
        
        // find kernel
        findBoundaryMinMaxKernel = computeShaderObj.FindKernel("FindBoundaryMinMax");
        
        // set up compute shader
        computeShaderObj.SetBuffer(findBoundaryMinMaxKernel, "boundaryPositions", boundaryPositionsBuffer);
        computeShaderObj.SetBuffer(findBoundaryMinMaxKernel, "boundaryBB", boundaryBBBuffer);
        
        // dispatch kernel
        computeShaderObj.Dispatch(findBoundaryMinMaxKernel, 1, 1, 1);
        
        // fetch data
        boundaryMinMax.Initialize();
        boundaryBBBuffer.GetData(boundaryMinMax);
        
        // print($"boundary min: {boundaryMinMax[0].min}, max {boundaryMinMax[0].max}");
    }

    private void SetupMeshData()
    {
        var number = numberOfObjects;
        //print(Application.dataPath);
        string filePath = Application.dataPath + "/TetModel/";
        LoadTetModel.LoadData(filePath + modelName, gameObject);
        List<List<string>> csvData = ExporterAndImporter.ReadCSVFile(csv_file);
        
        // if(number > csvData.Count) // exit app
        // {
        //     throw new Exception("cc"); 
        //     //  UnityEditor.EditorApplication.isPlaying = false;
        // } 

      
        indicies = new Vector2Int[number];
        int st_index = 0;

        var _Positions = LoadTetModel.positions.ToArray();            
        var _triangles = LoadTetModel.triangles;
        var _triArray = LoadTetModel.triangleArr.ToArray();

        Positions = new Vector3[number * LoadTetModel.positions.Count];
        Velocities = new Vector3[number * LoadTetModel.positions.Count];
        triangles = new List<Triangle>(new Triangle[number * LoadTetModel.triangles.Count]);
        triArray = new int[number * LoadTetModel.triangleArr.Count];


        // Initialize movement offset
        movementDirection = new Vector3[number];
        randomVelocity = new Vector3 [number];

         for(int i=0;i<number;i++){
            int PosOffset = i * LoadTetModel.positions.Count;
            int VelocitiesOffset = i * LoadTetModel.positions.Count;
            
            List<string> column = csvData[i];
            float x = float.Parse(column[0]);
            float y = float.Parse(column[1]);
            float z = float.Parse(column[2]);
            Vector3 Offset = new Vector3(x, y, z);

             movementDirection[i] = new Vector3(
                // exact min and max coor of booundary 
                Random.Range(boundaryMinMax[0].min.x, boundaryMinMax[0].max.x),
                Random.Range(boundaryMinMax[0].min.y, boundaryMinMax[0].max.y),
                Random.Range(boundaryMinMax[0].min.z, boundaryMinMax[0].max.z)
            );
            
            for(int j=0;j<LoadTetModel.positions.Count;j++){                
                // Positions[j+PosOffset] = _Positions[j] + Offset;
                Positions[j+PosOffset] = _Positions[j] + movementDirection[i];
            }
            
            randomVelocity[i] = new Vector3(
                Random.Range(-.1f, .1f),
                Random.Range(-.1f, .1f),
                Random.Range(-.1f, .1f)
            );
            
            // print($"random velocity {randomVelocity[i]}");
            
            for(int j=0;j<LoadTetModel.positions.Count;j++)
            {
                Velocities[j + VelocitiesOffset] = randomVelocity[i] ;
            }

            int TriOffset = i * LoadTetModel.triangles.Count;
            for(int j=0;j<LoadTetModel.triangles.Count;j++){
                var t = _triangles[j];
                triangles[j+TriOffset] = new Triangle(t.vertices[0] + PosOffset, t.vertices[1] + PosOffset, t.vertices[2] + PosOffset);
            }

            int TriArrOffset = i * LoadTetModel.triangleArr.Count;
            for(int j=0;j<LoadTetModel.triangleArr.Count;j++){
                triArray[j+TriArrOffset] = _triArray[j] + PosOffset;
            }

            indicies[i] = new Vector2Int(st_index, st_index + LoadTetModel.positions.Count);
            st_index += LoadTetModel.positions.Count;

        }        

        nodeCount = Positions.Length;
        triCount = triangles.Count; 
        vDataArray = new vertData[nodeCount];

        for (int i = 0; i < nodeCount; i++)
        {
            vDataArray[i] = new vertData();
            vDataArray[i].pos = Positions[i];
            vDataArray[i].norms = Vector3.zero;
            vDataArray[i].uvs = Vector3.zero;
        }

        int triBuffStride = sizeof(int);
        triBuffer = new ComputeBuffer(triArray.Length, triBuffStride, ComputeBufferType.Default);

        int vertsBuffstride = 8 * sizeof(float);
        vertsBuff = new ComputeBuffer(vDataArray.Length, vertsBuffstride, ComputeBufferType.Default);
        LoadTetModel.ClearData();

    }

    private void SetupShader()
    {
        material.SetBuffer(Shader.PropertyToID("vertsBuff"), vertsBuff);
        material.SetBuffer(Shader.PropertyToID("triBuff"), triBuffer);
    }

     private void SetBuffData()
    {
        vertsBuff.SetData(vDataArray);
        triBuffer.SetData(triArray);

        Vector3 translation = transform.position;
        Vector3 scale = this.transform.localScale;
        Quaternion rotationeuler = transform.rotation;
        Matrix4x4 trs = Matrix4x4.TRS(translation, rotationeuler, scale);
        material.SetMatrix("TRSMatrix", trs);
        material.SetMatrix("invTRSMatrix", trs.inverse);
    }

    private void SetupMeshComputeBuffer()
    {
        positionsBuffer = new ComputeBuffer(nodeCount, sizeof(float) * 3);
        positionsBuffer.SetData(Positions);

        velocitiesBuffer = new ComputeBuffer(nodeCount, sizeof(float) * 3);
        velocitiesBuffer.SetData(Velocities);
        
        randomVelocitiesBuffer = new ComputeBuffer(numberOfObjects, sizeof(float) * 3);
        randomVelocitiesBuffer.SetData(randomVelocity);

        List<MTriangle> initTriangle = new List<MTriangle>();  //list of triangle cooresponding to node 
        List<int> initTrianglePtr = new List<int>(); //contain a group of affectd triangle to node
     
        Dictionary<int, List<int>> nodeTriangles = new Dictionary<int, List<int>>();
        for (int triIndex = 0; triIndex < triangles.Count; triIndex++)
        {
            Triangle tri = triangles[triIndex];
            for (int vertexIndex = 0; vertexIndex < 3; vertexIndex++)
            {
                int vertex = tri.vertices[vertexIndex];
                if (!nodeTriangles.ContainsKey(vertex))
                {
                    nodeTriangles[vertex] = new List<int>();
                }
                nodeTriangles[vertex].Add(triIndex);
            }
        }
        
        initTrianglePtr.Add(0);
        for (int i = 0; i < nodeCount; i++)
        {
            if (nodeTriangles.TryGetValue(i, out List<int> triangleIndexes))
            {
                foreach (int triIndex in triangleIndexes)
                {
                    Triangle tri = triangles[triIndex];
                    MTriangle tmpTri = new MTriangle { v0 = tri.vertices[0], v1 = tri.vertices[1], v2 = tri.vertices[2] };
                    initTriangle.Add(tmpTri);
                }
            }
            initTrianglePtr.Add(initTriangle.Count);
        }


        triangleBuffer = new ComputeBuffer(initTriangle.Count, (sizeof(int) * 3));
        triangleBuffer.SetData(initTriangle.ToArray());

        triangleIndicesBuffer = new ComputeBuffer(initTrianglePtr.Count, sizeof(int));
        triangleIndicesBuffer.SetData(initTrianglePtr.ToArray());

        // print("node count: " + nodeCount);
    }

    private void SetupCollisionComputeBuffer()
    {
        bbOctree = new OctreeData[numberOfObjects * octree_size];
        collisionBox = new int[numberOfObjects * octree_size];
        nodePositions = new Vector3[nodeCount];
        
        bbMinMaxBuffer = new ComputeBuffer(numberOfObjects, sizeof(float) * 6);
        objectIndexBuffer = new ComputeBuffer(numberOfObjects, sizeof(int) * 2); 
        objectIndexBuffer.SetData(indicies);
        
        
        bbOctreeBuffer = new ComputeBuffer(numberOfObjects * octree_size, sizeof(float) * 12);
        octreeIndicesBuffer = new ComputeBuffer(octreeIndices.Count, sizeof(int));
        octreeIndicesBuffer.SetData(octreeIndices);
        
        relationVertexAABBBuffer = new ComputeBuffer(nodeCount, sizeof(int));
        relationVertexAABB = new int[nodeCount];
        relationVertexAABB.Initialize();
        relationVertexAABBBuffer.SetData(relationVertexAABB);
        
        collisionBoxBuffer = new ComputeBuffer(numberOfObjects * octree_size, sizeof(int));
        collisionBox.Initialize();
        collisionBoxBuffer.SetData(collisionBox);

        collisionNode = new Vector2Int[nodeCount];
        collisionNodeBuffer = new ComputeBuffer(nodeCount, sizeof(int) * 2);
        collisionNode.Initialize();
        collisionNodeBuffer.SetData(collisionNode);
        
        nodeInsideBox = new int[nodeCount];
        nodeInsideBox.Initialize();
        nodeInsideBoxBuffer = new ComputeBuffer(nodeCount, sizeof(int));
        nodeInsideBoxBuffer.SetData(nodeInsideBox);
        
        nodeCounterData = new int[1];
        nodeIndex = new int[nodeCount];
        allNodeIndex = new int[nodeCount];

        // nodeIndex.Initialize();
        for (int i = 0; i < nodeCount; i++)
        {
            nodeIndex[i] = i;
        }
        // Initialize the index buffer with enough space to store all possible indices
        nodeIndexBuffer = new ComputeBuffer(nodeCount, sizeof(int));
        // nodeIndexBuffer.SetData(nodeIndex);
        allNodeIndexBuffer = new ComputeBuffer(nodeCount, sizeof(int));

        nodeProcessedUniqueBuffer = new ComputeBuffer(nodeCount, sizeof(int));
        int[] nodeProcessed = new int[nodeCount];
        nodeProcessedUniqueBuffer.SetData(nodeProcessed);
        
        nodeUniqueBuffer = new ComputeBuffer(nodeCount, sizeof(int));
        int[] nodeUniqe = new int[nodeCount];
        nodeUniqe.Initialize();
        nodeProcessedUniqueBuffer.SetData(nodeUniqe);
        
        nodeCounterBuffer = new ComputeBuffer(1, sizeof(int)); // Initialize the counter buffer with a single integer
        int[] initialNodeCounter = new int[] { 0 }; // Set the initial value of the counter to 0
        nodeCounterBuffer.SetData(initialNodeCounter);
        allNodeCounterBuffer = new ComputeBuffer(1, sizeof(int)); 
        int[] initialAllNodeCounter = new int[] { 0 }; 
        allNodeCounterBuffer.SetData(initialAllNodeCounter);
        

        collisionTesting = new int[1];
        collisionTesting.Initialize();
        collisionTestingBuffer = new ComputeBuffer(1, sizeof(int));
        collisionTestingBuffer.SetData(collisionTesting);
        
        
        indexData = new int[nodeCount];
        indexData.Initialize();

        collidedNodeFlagBuffer = new ComputeBuffer(1, sizeof(int));
    }

    private void SetupKernel()
    {
        //for rendering
        computeVerticesNormal = computeShaderObj.FindKernel("computeVerticesNormal");
        updatePosKernel = computeShaderObj.FindKernel("UpdatePosKernel");
        findBBMinMaxKernel = computeShaderObj.FindKernel("FindBBMinMax");
        CollisionBoundaryHandlingKernel = computeShaderObj.FindKernel("CollisionBoundaryHandling");
        ImplementOctreeKernel = computeShaderObj.FindKernel("ImplementOctree");
        CheckBBCollisionKernel = computeShaderObj.FindKernel("CheckBBCollision");
        UpdateVertexAndBoxRelationKernel = computeShaderObj.FindKernel("UpdateVertexBoundingRelation"); 
        rmCollisionsBoxKernel = computeShaderObj.FindKernel("removeCollisionsBox");
        rmNodeCollisionKernel = computeShaderObj.FindKernel("RemoveNodeCollision");
        rmNodeCounterBufferKernel = computeShaderObj.FindKernel("RemoveNodeCounter");
        UpdateNodeInsideBoxKernel = computeShaderObj.FindKernel("UpdateBoxContainVertex");
        CollisionResponseKernel = computeShaderObj.FindKernel("CollisionResponse");
        UpdateCollidedNodeFlagKernel = computeShaderObj.FindKernel("UpdateCollidedNodeFlag");
        CollisionDetectionResponseKernel = computeShaderObj.FindKernel("CollisionDetectionResponse");
    }

    
    private IEnumerator  SetupComputeShader()
    {
        //send uniform data for kernels in compute shader
        computeShaderObj.SetFloat("dt", dt);
        computeShaderObj.SetVector("gravity", gravity);
        computeShaderObj.SetFloat("invMass", invMass);
        computeShaderObj.SetInt("triCount", triCount);
        computeShaderObj.SetInt("nodeCount", nodeCount);
        computeShaderObj.SetInt("numberObj", numberOfObjects);
        computeShaderObj.SetInt("octree_size", octree_size);
        computeShaderObj.SetFloat("collisionDamping", collisionDamping);
        computeShaderObj.SetFloat("separationBias", separationBias);
        computeShaderObj.SetFloat("restitution", restitution);
        computeShaderObj.SetFloat("dampingPosition", dampingPosition);
        computeShaderObj.SetFloat("dampingVelocity", dampingVelocity);
        computeShaderObj.SetFloat("impulseFactor", impulseFactor);
        
        
        // computeVerticesNormal
        computeShaderObj.SetBuffer(computeVerticesNormal, "Positions", positionsBuffer);
        computeShaderObj.SetBuffer(computeVerticesNormal, "Triangles", triangleBuffer);
        computeShaderObj.SetBuffer(computeVerticesNormal, "TrianglePtr", triangleIndicesBuffer);
        computeShaderObj.SetBuffer(computeVerticesNormal, "vertsBuff", vertsBuff); //passing to rendering
        
        // UpdatePosKernel
        computeShaderObj.SetBuffer(updatePosKernel, "Positions", positionsBuffer);
        computeShaderObj.SetBuffer(updatePosKernel, "Velocities", velocitiesBuffer);
        computeShaderObj.SetBuffer(updatePosKernel, "ObjectIndex", objectIndexBuffer);
        computeShaderObj.SetBuffer(updatePosKernel, "vertsBuff", vertsBuff); //passing to rendering
        
        
        // findBBMinMaxKernel
        computeShaderObj.SetBuffer(findBBMinMaxKernel, "bbMinMax", bbMinMaxBuffer);
        computeShaderObj.SetBuffer(findBBMinMaxKernel, "ObjectIndex", objectIndexBuffer);
        computeShaderObj.SetBuffer(findBBMinMaxKernel, "Positions", positionsBuffer);
        
        // CollisionHandlingKernel
        computeShaderObj.SetBuffer(CollisionBoundaryHandlingKernel, "Positions", positionsBuffer);
        computeShaderObj.SetBuffer(CollisionBoundaryHandlingKernel, "Velocities", velocitiesBuffer);
        computeShaderObj.SetBuffer(CollisionBoundaryHandlingKernel, "ObjectIndex", objectIndexBuffer);
        computeShaderObj.SetBuffer(CollisionBoundaryHandlingKernel, "boundaryBB", boundaryBBBuffer);
        computeShaderObj.SetBuffer(CollisionBoundaryHandlingKernel, "vertsBuff", vertsBuff); //passing to rendering
        computeShaderObj.SetBuffer(CollisionBoundaryHandlingKernel, "bbMinMax", bbMinMaxBuffer);
        
        // UpdateTriAABBRelation
        computeShaderObj.SetBuffer(UpdateVertexAndBoxRelationKernel, "Positions", positionsBuffer);
        computeShaderObj.SetBuffer(UpdateVertexAndBoxRelationKernel, "bbOctree", bbOctreeBuffer);
        computeShaderObj.SetBuffer(UpdateVertexAndBoxRelationKernel, "relationVertexAABB", relationVertexAABBBuffer);
        computeShaderObj.SetBuffer(UpdateVertexAndBoxRelationKernel, "octreeIndices", octreeIndicesBuffer);
        computeShaderObj.SetBuffer(UpdateVertexAndBoxRelationKernel, "collisionBox", collisionBoxBuffer);
        
        // rmCollisionsBox
        computeShaderObj.SetBuffer(rmCollisionsBoxKernel, CollisionBox, collisionBoxBuffer);
        computeShaderObj.SetBuffer(rmCollisionsBoxKernel, "collisionNode", collisionNodeBuffer);
        computeShaderObj.SetBuffer(rmCollisionsBoxKernel, "collisionTesting", collisionTestingBuffer);
        
        // RemoveNodeKernel
        computeShaderObj.SetBuffer(rmNodeCollisionKernel, "nodeInsideBox", nodeInsideBoxBuffer);
        computeShaderObj.SetBuffer(rmNodeCollisionKernel, "collisionNode", collisionNodeBuffer);
        computeShaderObj.SetBuffer(rmNodeCollisionKernel, "allNodeIndex", allNodeIndexBuffer);
        
        // RemoveTriCollision
        computeShaderObj.SetBuffer(rmNodeCounterBufferKernel, "nodeCounter", nodeCounterBuffer);
        computeShaderObj.SetBuffer(rmNodeCounterBufferKernel, "allNodeCounter", allNodeCounterBuffer);
        computeShaderObj.SetBuffer(rmNodeCounterBufferKernel, "nodeProcessedUnique", nodeProcessedUniqueBuffer);
        computeShaderObj.SetBuffer(rmNodeCounterBufferKernel, "collidedNodeFlag", collidedNodeFlagBuffer);
        
        // CheckBBCollision
        computeShaderObj.SetBuffer(CheckBBCollisionKernel, "octreeIndices", octreeIndicesBuffer);
        computeShaderObj.SetBuffer(CheckBBCollisionKernel, "collisionBox", collisionBoxBuffer);
        computeShaderObj.SetBuffer(CheckBBCollisionKernel, "bbOctree", bbOctreeBuffer);
        
        
        // UpdateNodeInsideBoxKernel
        computeShaderObj.SetBuffer(UpdateNodeInsideBoxKernel, "bbOctree", bbOctreeBuffer);
        computeShaderObj.SetBuffer(UpdateNodeInsideBoxKernel, "relationVertexAABB", relationVertexAABBBuffer);
        computeShaderObj.SetBuffer(UpdateNodeInsideBoxKernel, "collisionBox", collisionBoxBuffer);
        computeShaderObj.SetBuffer(UpdateNodeInsideBoxKernel, "nodeCounter", nodeCounterBuffer);
        computeShaderObj.SetBuffer(UpdateNodeInsideBoxKernel, "nodeIndex", nodeIndexBuffer);
        computeShaderObj.SetBuffer(UpdateNodeInsideBoxKernel, "nodeProcessedUnique", nodeProcessedUniqueBuffer);
        computeShaderObj.SetBuffer(UpdateNodeInsideBoxKernel, "nodeInsideBox", nodeInsideBoxBuffer);
        
        // UpdateCollidedNodeFlagKernel
        computeShaderObj.SetBuffer(UpdateCollidedNodeFlagKernel, "nodeCounter", nodeCounterBuffer);
        // computeShaderObj.SetBuffer(UpdateCollidedNodeFlagKernel, "nodeIndex", nodeIndexBuffer);
        // computeShaderObj.SetBuffer(UpdateCollidedNodeFlagKernel, "allNodeCounter", allNodeCounterBuffer);
        computeShaderObj.SetBuffer(UpdateCollidedNodeFlagKernel, "allNodeIndex", allNodeIndexBuffer);
        computeShaderObj.SetBuffer(UpdateCollidedNodeFlagKernel, "collisionTesting", collisionTestingBuffer);
        // computeShaderObj.SetBuffer(UpdateCollidedNodeFlagKernel, "octreeIndices", octreeIndicesBuffer);
        computeShaderObj.SetBuffer(UpdateCollidedNodeFlagKernel, "bbMinMax", bbMinMaxBuffer);
        // computeShaderObj.SetBuffer(UpdateCollidedNodeFlagKernel, "Positions", positionsBuffer);
        
        
        // CollisionResponseKernel
        computeShaderObj.SetBuffer(CollisionResponseKernel, "Positions", positionsBuffer);
        computeShaderObj.SetBuffer(CollisionResponseKernel, "Velocities", velocitiesBuffer);
        computeShaderObj.SetBuffer(CollisionResponseKernel, "bbOctree", bbOctreeBuffer);
        computeShaderObj.SetBuffer(CollisionResponseKernel, "randomVelocities", randomVelocitiesBuffer);
        computeShaderObj.SetBuffer(CollisionResponseKernel, "bbMinMax", bbMinMaxBuffer);
        computeShaderObj.SetBuffer(CollisionResponseKernel, "collisionTesting", collisionTestingBuffer);
        computeShaderObj.SetBuffer(CollisionResponseKernel, "nodeIndex", nodeIndexBuffer);
        computeShaderObj.SetBuffer(CollisionResponseKernel, "nodeCounter", nodeCounterBuffer);
        computeShaderObj.SetBuffer(CollisionResponseKernel, "collidedNodeFlag", collidedNodeFlagBuffer);
        
        
        // CollisionDetectionResponse
        computeShaderObj.SetBuffer(CollisionDetectionResponseKernel, "Positions", positionsBuffer);
        computeShaderObj.SetBuffer(CollisionDetectionResponseKernel, "Velocities", velocitiesBuffer);
        computeShaderObj.SetBuffer(CollisionDetectionResponseKernel, "bbOctree", bbOctreeBuffer);
        computeShaderObj.SetBuffer(CollisionDetectionResponseKernel, "randomVelocities", randomVelocitiesBuffer);
        computeShaderObj.SetBuffer(CollisionDetectionResponseKernel, "bbMinMax", bbMinMaxBuffer);
        computeShaderObj.SetBuffer(CollisionDetectionResponseKernel, "collidedNodeFlag", collidedNodeFlagBuffer);
        computeShaderObj.SetBuffer(CollisionDetectionResponseKernel, "allNodeCounter", allNodeCounterBuffer);
        computeShaderObj.SetBuffer(CollisionDetectionResponseKernel, "allNodeIndex", allNodeIndexBuffer);
        // computeShaderObj.SetBuffer(CollisionDetectionResponseKernel, "collisionTesting", collisionTestingBuffer);
        
        // ImplementOctree
        computeShaderObj.SetBuffer(ImplementOctreeKernel, "bbMinMax", bbMinMaxBuffer);
        computeShaderObj.SetBuffer(ImplementOctreeKernel, "bbOctree", bbOctreeBuffer);
        computeShaderObj.Dispatch(ImplementOctreeKernel, Mathf.CeilToInt(numberOfObjects / 1024.0f), 1, 1);
        bbOctreeBuffer.GetData(bbOctree);
        yield return new WaitForSeconds(0.01f);
        
        computeShaderObj.Dispatch(UpdateVertexAndBoxRelationKernel, Mathf.CeilToInt(nodeCount / 1024.0f), 1, 1);
        
        relationVertexAABBBuffer.GetData(relationVertexAABB);
        // for(int n = 0; n < relationVertexAABB.Length; n++) 
        // {
        //      print($"relation vertex and bounding {n} {relationVertexAABB[n]}");
        // }
    }

    void Update()
    {
        DispatchComputeShader();
        renderObject();
        GetDataToCPU();


        deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;
    }
    void  DispatchComputeShader()
    {
        var numGroups_Obj =  Mathf.CeilToInt(numberOfObjects / 32f);
        var numGroups_octIndices =  Mathf.CeilToInt(octreeIndices.Count / 32f);
        var numGroups_nodeCounts =  Mathf.CeilToInt(nodeCount / 32f);
        

        //compute normal for rendering
        computeShaderObj.Dispatch(computeVerticesNormal, (int)Mathf.Ceil(nodeCount / 1024.0f), 1, 1);
        computeShaderObj.Dispatch(rmCollisionsBoxKernel, Mathf.CeilToInt(numberOfObjects * octree_size / 1024f), 1, 1);
        computeShaderObj.Dispatch(rmNodeCollisionKernel, Mathf.CeilToInt(nodeCount / 1024.0f), 1, 1);
        computeShaderObj.Dispatch(rmNodeCounterBufferKernel, Mathf.CeilToInt(nodeCount / 1024f), 1, 1);
        computeShaderObj.Dispatch(updatePosKernel, Mathf.CeilToInt(nodeCount / 1024.0f), 1, 1);
        computeShaderObj.Dispatch(findBBMinMaxKernel, Mathf.CeilToInt(numberOfObjects / 1024f), 1, 1);
        computeShaderObj.Dispatch(ImplementOctreeKernel, Mathf.CeilToInt(numberOfObjects / 1024f), 1, 1);
        computeShaderObj.Dispatch(CollisionBoundaryHandlingKernel, Mathf.CeilToInt(nodeCount / 1024f), 1, 1);
        computeShaderObj.Dispatch(CheckBBCollisionKernel, numGroups_octIndices, numGroups_octIndices, 1); 
        // computeShaderObj.Dispatch(UpdateNodeInsideBoxKernel,  Mathf.CeilToInt(nodeCount / 1024.0f), 1, 1);
        
        // Dispatch the compute shader with the calculated thread group size
        
        int[] _nodeCounter = new int[1];
        int[] _allNodeCounter = new int[1];
        nodeCounterBuffer.GetData(_nodeCounter);
        int validNodeCounter =  _nodeCounter[0] == 0? 1 : _nodeCounter[0];
        
        
        // print($"node counter {_nodeCounter[0]}");
        // if (_nodeCounter[0] > 0)
        // {
        //     // Retrieve the indices of valid nodes
        //     int[] nodeIndices = new int[_nodeCounter[0]];
        //     nodeIndexBuffer.GetData(nodeIndices, 0, 0, _nodeCounter[0]);
        //     // print("Node Indices: " + string.Join(", ", nodeIndices));
        // }
        
        
        // computeShaderObj.Dispatch(UpdateCollidedNodeFlagKernel, Mathf.CeilToInt(nodeCount / 1024.0f), 1, 1);
        
        // computeShaderObj.Dispatch(CollisionResponseKernel, numGroups_Obj, numGroups_Obj, 1);
        computeShaderObj.Dispatch(CollisionDetectionResponseKernel, Mathf.CeilToInt(nodeCount / 1024.0f), 1, 1);
    }
    
    void renderObject()
    {
        Bounds bounds = new Bounds(Vector3.zero, Vector3.one * 10000);
        material.SetPass(0);
        Graphics.DrawProcedural(material, bounds, MeshTopology.Triangles, triArray.Length,
            1, null, null, ShadowCastingMode.On, true, gameObject.layer);
    }

    private void GetDataToCPU()
    {
        triIntersectionCounting.Clear();
        
        if (debugBox)
        {
            bbOctreeBuffer.GetData(bbOctree);
            
            collisionBoxBuffer.GetData(collisionBox);
            
            
            // for (int c = 0; c < collisionBox.Length; c++)
            // {
            //     if (collisionBox[c] == 1)  
            //     {
            //         print($"box {c} ::  {collisionBox[c]}");
            //     }
            // }
            
            
            
            
            
        }

        if (debugNode)
        {
            nodeInsideBoxBuffer.GetData(nodeInsideBox);
            positionsBuffer.GetData(nodePositions);
            
            collisionNodeBuffer.GetData(collisionNode);

            for (int c = 0; c < collisionNode.Length; c++)
            {
                
                if (collisionNode[c].x == 1 || collisionNode[c].y == 1) // Check collision on x or y
                {
                    if (!triIntersectionCounting.Contains(c))
                        triIntersectionCounting.Add(c);
                }
            }
        }
    }

    

    void OnDrawGizmos()
    {
        if(collisionBox != null && debugBox)
        {
            for (int i = 0; i < collisionBox.Length; i++)
            {
                if (boxLv0 || boxLv1 || boxLv2)
                {
                    if(collisionBox[i] == 1)
                    {
                      
                        if(boxLv0) Gizmos.color = Color.red;        
                        if(boxLv1) Gizmos.color = Color.green;
                        if(boxLv2) Gizmos.color = Color.blue;
        
                        int id = collisionBox[i];
                        Gizmos.DrawWireCube(bbOctree[i].center, bbOctree[i].max - bbOctree[i].min);
                        
                    }
                }
        
                
            }
        }
        
        if (debugNode && nodeInsideBox !=null)
        {
            for (int i = 0; i < nodeInsideBox.Length; i++)
            {
                Gizmos.color = Color.red;
                if(nodeInsideBox[i] ==1)
                    Gizmos.DrawSphere(nodePositions[i], 0.005f);
            }
            
            // Gizmos.color =Color.yellow;
            // for (int i = 0; i < triIntersectionCounting.Count; i++)
            // {
            //     Gizmos.DrawSphere(nodePositions[triIntersectionCounting[i]], 0.01f);  
            // }
            
        }

        if (wholeNodes && nodePositions != null)
        {
            for (int i = 0; i < nodePositions.Length; i++)
            {
                Gizmos.color = Color.red;
                Gizmos.DrawSphere(nodePositions[i], 0.005f);
            }
        }
    }
    
    
    private void OnGUI()
    {
        int w = Screen.width, h = Screen.height;
        GUIStyle style = new GUIStyle();
        style.alignment = TextAnchor.UpperLeft;
        style.fontSize = h * 2 / 50;
        style.normal.textColor = Color.yellow;

        // Rect rect = new Rect(20, 40, w, h * 2 / 100);
        // string text = string.Format("num. Obj :: " + numberOfObjects);
        // GUI.Label(rect, text, style);


        deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;
        float fps = 1.0f / deltaTime;
        float ms = deltaTime * 1000.0f;


        float rectWidth = 200f;
        float rectHeight = 100f;

        // Calculate the position to place the second rectangle at the top middle of the screen
        // float topX = (Screen.width / 2) - (rectWidth / 2) ;
        // float topY = 0f; // Y position at the top of the screen


         // Calculate the position to place the third rectangle at the bottom middle of the screen
        float bottomX = (Screen.width / 2) - (rectWidth / 2);
        float bottomY = Screen.height - rectHeight; // Y position at the bottom of the screen

        // Create the rectangles
        // Rect rectNumObj = new Rect(topX - 50 , topY, rectWidth, rectHeight);
        // Rect rectFPS = new Rect(topX - 54 , topY + 20, rectWidth, rectHeight);
        Rect rectMeshInfo = new Rect(bottomX - rectWidth + 30, bottomY + 25, rectWidth, rectHeight);
        Rect rectNumObj = new Rect(bottomX - rectWidth + 30, bottomY + 45, rectWidth, rectHeight);
        Rect rectFPS = new Rect(bottomX - rectWidth + 30, bottomY + 65, rectWidth, rectHeight);

        // Draw the rectangles
        string originalString = modelName;
        string[] splitModelName = originalString.Split('.');

        // string stMeshInfo = "Node Counts: " + nodeCount.ToString("N0") + ", Triangle Counts: "+ triCount.ToString("N0");
        string stMeshInfo = "Node Counts: " + nodeCount.ToString("N0");
        string stNumObj = "Number of Objects: " + numberOfObjects.ToString() + " "+ splitModelName[0] + " Models";
        
        // string stFPS = string.Format(" FPS :: "+  Mathf.Round(fps * 10.0f) * 0.1f  + "(" + Mathf.Round(ms * 10.0f) * 0.1f + "ms)");
        string stFPS = string.Format("FPS: "+  Mathf.Round(fps * 10.0f) * 0.1f +", Max: " + Mathf.Round(ms * 10.0f) * 0.1f + "ms");

        GUI.Label(rectMeshInfo, stMeshInfo, style);
        GUI.Label(rectNumObj, stNumObj, style);
        GUI.Label(rectFPS, stFPS, style);
            
    }

    private void OnDestroy() 
    {
        if (enabled)
        {
            vertsBuff.Dispose();
            triBuffer.Dispose();
            positionsBuffer.Dispose();
            velocitiesBuffer.Dispose();
            triangleBuffer.Dispose();
            triangleIndicesBuffer.Dispose();
            boundaryBBBuffer.Dispose();
            boundaryPositionsBuffer.Dispose();
            bbMinMaxBuffer.Dispose();
            objectIndexBuffer.Dispose();
            bbOctreeBuffer.Dispose();
            octreeIndicesBuffer.Dispose();
            randomVelocitiesBuffer.Dispose();
            relationVertexAABBBuffer.Dispose();
            collisionBoxBuffer.Dispose();
            collisionNodeBuffer.Dispose();
            collisionTestingBuffer.Dispose();
            nodeInsideBoxBuffer.Dispose();
            nodeIndexBuffer.Dispose();
            nodeCounterBuffer.Dispose();
            nodeProcessedUniqueBuffer.Dispose();
            collidedNodeFlagBuffer.Dispose();
            
            if (SetupComputeShader() != null)
                StopCoroutine(SetupComputeShader());
            
   
            
        }    
    }
}

