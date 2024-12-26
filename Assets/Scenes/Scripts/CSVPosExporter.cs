using ExporterImporter;
using System.Collections;
using System.Collections.Generic;
using Octree;
using UnityEngine;
using UnityEngine.Serialization;

public class CSVPosExporter : MonoBehaviour
{
    [Header("Number model")]
    public int number_object = 1;
    // public GameObject clone_object;
    [SerializeField] GameObject boundary;

    [Header("Random Range")]
    // public Vector3 rangeMin = new Vector3(-10f, 0f, 0f);
    // public Vector3 rangeMax = new Vector3(10f, 10f, 20f);

    private List<Vector3> object_position = new List<Vector3>();
    private BoundingBox boundaryBB;

    void Start()
    {
        GenerateObjectPosition();
        ExportPosition();
    }

    private void GenerateObjectPosition()
    {
        HashSet<Vector3> generatedPositions = new HashSet<Vector3>();
        
        
        Vector3[] vertices;

        vertices = boundary.GetComponent<MeshFilter>().mesh.vertices;
        boundaryBB.min = boundary.transform.TransformPoint(vertices[0]);
        boundaryBB.max = boundary.transform.TransformPoint(vertices[0]);


        for (int i = 0; i < vertices.Length; i++)
        {
            Vector3 allVerts = boundary.transform.TransformPoint(vertices[i]);

            boundaryBB.min.x = Mathf.Min(boundaryBB.min.x, allVerts.x);
            boundaryBB.min.y = Mathf.Min(boundaryBB.min.y, allVerts.y);
            boundaryBB.min.z = Mathf.Min(boundaryBB.min.z, allVerts.z);

            boundaryBB.max.x = Mathf.Max(boundaryBB.max.x, allVerts.x);
            boundaryBB.max.y = Mathf.Max(boundaryBB.max.y, allVerts.y);
            boundaryBB.max.z = Mathf.Max(boundaryBB.max.z, allVerts.z);
        }

        // for (int i = 0; i < number_object; i++)
        // {
        //     Vector3 randomPosition;
        //
        //     do
        //     {
        //         // Generate random position within the specified range
        //         float x = Random.Range(rangeMin.x, rangeMax.x);
        //         float y = Random.Range(rangeMin.y, rangeMax.y);
        //         float z = Random.Range(rangeMin.z, rangeMax.z);
        //
        //         randomPosition = new Vector3(x, y, z);
        //
        //         //Instantiate(clone_object, randomPosition, transform.rotation);
        //         object_position.Add(randomPosition);
        //
        //     } while (generatedPositions.Contains(randomPosition));
        // }
        
        
        // for (int i = 0; i < number_object; i++)
        // {
        //     Vector3 randomPosition;
        //     bool positionIsValid;
        //
        //     do
        //     {
        //         // Generate random position within the specified range
        //         // float x = Random.Range(rangeMin.x, rangeMax.x);
        //         // float y = Random.Range(rangeMin.y, rangeMax.y);
        //         // float z = Random.Range(rangeMin.z, rangeMax.z);
        //
        //         var x =Random.Range(boundaryBB.min.x, boundaryBB.max.x);
        //         var y =Random.Range(boundaryBB.min.y, boundaryBB.max.y);
        //         var z = Random.Range(boundaryBB.min.z, boundaryBB.max.z);
        //
        //         randomPosition = new Vector3(x, y, z);
        //         positionIsValid = true;
        //
        //         // Check if the new position is too close to any existing object
        //         foreach (var existingPos in generatedPositions)
        //         {
        //             if (Vector3.Distance(existingPos, randomPosition) < 1*2) // minDistance is the minimum allowed distance to avoid overlap
        //             {
        //                 positionIsValid = false;
        //                 break;
        //             }
        //         }
        //
        //         // If valid, add it to the generated positions
        //         if (positionIsValid)
        //         {
        //             generatedPositions.Add(randomPosition);
        //             object_position.Add(randomPosition);
        //             //Instantiate(clone_object, randomPosition, transform.rotation);
        //         }
        //
        //     } while (object_position.Count == number_object); // Keep trying until a valid position is found
        // }
        
        
        for (int i = 0; i < number_object; i++)
        {
            Vector3 randomPosition;
            // bool positionIsValid;

            // Generate random position within the specified range
            do
            {
                float x = Random.Range(boundaryBB.min.x, boundaryBB.max.x);
                float y = Random.Range(boundaryBB.min.y, boundaryBB.max.y);
                float z = Random.Range(boundaryBB.min.z, boundaryBB.max.z);

                randomPosition = new Vector3(x, y, z);
                // positionIsValid = true;

                // Check if the new position is too close to any existing object
                foreach (var existingPos in generatedPositions)
                {
                    if (Vector3.Distance(existingPos, randomPosition) < 1 * 2) // minDistance is the minimum allowed distance to avoid overlap
                    {
                        // positionIsValid = false;
                        break;
                    }
                }
            } while (object_position.Count == number_object); // Keep generating a new position until a valid one is found

            generatedPositions.Add(randomPosition);
            object_position.Add(randomPosition);
            // Instantiate(clone_object, randomPosition, transform.rotation);
        }

    }

    private void ExportPosition()
    {
        ExporterAndImporter exporter = new ExporterAndImporter(object_position.ToArray());
        exporter.ExportPositionsToExcel();
    }
}
