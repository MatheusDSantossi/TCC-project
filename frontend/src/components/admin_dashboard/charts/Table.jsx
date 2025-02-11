import { styled, TableCell, tableCellClasses } from "@mui/material";
import Box from "@mui/material/Box";
import { DataGrid } from "@mui/x-data-grid";
import { IconButton } from "@mui/material";
import { MdEdit } from "react-icons/md";
import { useState } from "react";
import AxiosInstance from "../../AxiosInstance";
import ProductEdit from "../../ProductEdit";
import { Diversity1Outlined } from "@mui/icons-material";

// const StyleTableCell = styled(TableCell)(({ theme }) => ({
//   [`&.${tableCellClasses.head}`]: {
//     backgroundColor: theme.palette.common.black,
//     color: theme.palette.common.white,
//     fontWeight: "bold",
//     fontSize: "1rem",
//     whiteSpace: "nowrap",
//     padding: "10px 16px",
//   },
//   [`&.${tableCellClasses.body}`]: {
//     fontSize: 14,
//   },
// }));

// const StyledTableRow = styled(TableRow)((({ theme }) => ({
//   '&:nth-of-type(odd)': {
//     backgroundColor: theme.palette.action.hover,
//   },
//   // Hide last border
//   '&:last-child td, &:last-child th': {
//     border: 0,
//   },
// })));

export default function ProductsTable({ products_data, onProductUpdate }) {
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);

  const [loading, setLoading] = useState(false);

  const handleEditClick = (product) => {
    console.log("Product: ", product);
    setSelectedProduct(product);
    setEditModalOpen(true);
    setLoading(true);
  };

  const handleSave = async (updatedProduct) => {
    try {
      const response = await AxiosInstance.put(
        `update-product/${updatedProduct.productId}/`,
        {
          productName: updatedProduct.productName,
          productBrand: updatedProduct.productBrand,
          productQuantityFormat: updatedProduct.productQuantityFormat,
          productSection: updatedProduct.productSection,
        }
      );
      console.log("Product updated successfully: ", updatedProduct);

      if (response.status === 201 || response.status === 200) {
        onProductUpdate();
      }
    } catch (error) {
      console.error("Error updating product: ", error);
    }
  };

  const handleClose = () => {
    // setLoading(false);
    setEditModalOpen(false);
  };

  const columns = [
    {
      field: "product_id",
      headerName: <strong>ID</strong>,
      width: 90,
    },
    {
      field: "name",
      headerName: <strong>Produto</strong>,
      width: 200,
    },
    {
      field: "brand",
      headerName: <strong>Marca</strong>,
      width: 150,
    },
    {
      field: "quantity",
      headerName: <strong>Formato da Quantidade</strong>,
      width: 180,
    },
    {
      field: "secao_produto",
      headerName: <strong>Seção</strong>,
      width: 100,
    },
    {
      field: "actions",
      headerName: <strong>Ações</strong>,
      width: 100,
      sortable: false,
      renderCell: (params) => (
        <IconButton
          onClick={() => handleEditClick(params.row)}
          size="small"
          color="primary"
        >
          <MdEdit color="#5D76D3" />
        </IconButton>
      ),
    },
  ];

  let rows = [];

  const products_dict = () => {
    for (const product of products_data) {
      rows.push(product);
    }

    return rows;
  };

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
    <div className="flex items-center justify-center w-full overflow-x-auto">
      <Box sx={{ height: 650, width: 900 }}>
        <DataGrid
          rows={products_data}
          columns={columns}
          getRowId={(row) => row.product_id}
          initialState={{
            pagination: {
              paginationModel: {
                pageSize: 10,
              },
            },
          }}
          pageSizeOptions={[10]}
          disableRowSelectionOnClick
        />
        <ProductEdit
          open={editModalOpen}
          onClose={handleClose}
          product={selectedProduct}
          onSave={handleSave}
        />
        
      </Box>
    </div>
  );
}
