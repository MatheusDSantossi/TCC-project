import Box from '@mui/material/Box';
import { DataGrid } from '@mui/x-data-grid';

const columns = [
    { 
      field: 'product_id', 
      headerName: <strong>ID</strong>, 
      width: 90,
    },
    {
      field: 'name',
      headerName: <strong>Product</strong>,
      width: 150,
    },
    {
      field: 'brand',
      headerName: <strong>Brand</strong>,
      width: 150,
    },
    {
      field: 'quantity',
      headerName: <strong>Quantity</strong>,
      width: 110,
    },
    {
      field: 'secao_produto',
      headerName: <strong>Section</strong>,
      width: 100,
    },
  ];
  
export default function DataGridDemo({ products_data }) {

    let rows = [];

    const products_dict = () => {
        for (const product of products_data) {
            rows.push(product);
        }

        return rows;
    }

    console.log("Product dict: ", products_dict()[0]);

    // const rows = [
    //     { id: 1, lastName: 'Snow', firstName: 'Jon', age: 14 },
    //     { id: 2, lastName: 'Lannister', firstName: 'Cersei', age: 31 },
    //     { id: 3, lastName: 'Lannister', firstName: 'Jaime', age: 31 },
    //     { id: 4, lastName: 'Stark', firstName: 'Arya', age: 11 },
    //     { id: 5, lastName: 'Targaryen', firstName: 'Daenerys', age: null },
    //     { id: 6, lastName: 'Melisandre', firstName: null, age: 150 },
    //     { id: 7, lastName: 'Clifford', firstName: 'Ferrara', age: 44 },
    //     { id: 8, lastName: 'Frances', firstName: 'Rossini', age: 36 },
    //     { id: 9, lastName: 'Roxie', firstName: 'Harvey', age: 65 },
    //   ];
      
    return (
        <Box sx={{ height: 400, width: 700 }}>
            <DataGrid 
            rows={products_data} 
            columns={columns} 
            getRowId={(row) => row.product_id}
            initialState={{
                pagination: {
                    paginationModel: {
                        pageSize: 5,
                    },
                },
            }}
            pageSizeOptions={[5]} 
            checkboxSelection
            disableRowSelectionOnClick
            />
        </Box>
    );
}
