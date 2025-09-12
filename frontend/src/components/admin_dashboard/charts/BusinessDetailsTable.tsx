import Box from "@mui/material/Box";
import { DataGrid } from "@mui/x-data-grid";
import { IconButton } from "@mui/material";
import { MdEdit } from "react-icons/md";
import { useState } from "react";
import AxiosInstance from "../../AxiosInstance";
import BusinessDetailEdit from "../../BusinessDetailEdit";

export default function BusinessDetailsTable({ business_detail_data, onBusinessDetailUpdate }) {
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [selectedBusinessDetail, setSelectedBusinessDetail] = useState(null);

  const handleEditClick = (businessDetail) => {
    console.log("BusinessDetail selected: ", businessDetail);
    setSelectedBusinessDetail(businessDetail);
    setEditModalOpen(true);
  };

  const handleSave = async (updatedBusinessDetail) => {
    console.log("Updated BusinessDetail: ", updatedBusinessDetail);
    try {
      const response = await AxiosInstance.put(
        `api/business-detail/${updatedBusinessDetail.id}/`,
        {
          id: updatedBusinessDetail.id,
          key: updatedBusinessDetail.key,
          value: updatedBusinessDetail.value,
        }
      );
      console.log("BusinessDetail updated successfully: ", updatedBusinessDetail);

      if (response.status === 201 || response.status === 200) {
        onBusinessDetailUpdate();
      }
    } catch (error) {
      console.error("Error updating BusinessDetail: ", error);
    }
  };

  const handleRefreshTable = () => {
    console.log("Entered to refresh table");
    onBusinessDetailUpdate();
  };

  const handleClose = () => {
    // setLoading(false);
    setEditModalOpen(false);
  };

  const columns = [
    {
      field: "id",
      headerName: <strong className="text-base">ID</strong>,
      width: 90,
      minWidth: 80,
    },
    {
      field: "key",
      headerName: <strong className="text-base">Título</strong>,
      minWidth: 200,
      flex: 2,
    },
    {
      field: "value",
      headerName: <strong className="text-base">Descrição</strong>,
      minWidth: 150,
      flex: 1,
    },
    {
      field: "actions",
      headerName: <strong className="text-base">Ações</strong>,
      minWidth: 100,
      flex: 0.5,
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

  // const rows = BusinessDetails_data;

  let rows = [];

  const BusinessDetails_dict = () => {
    for (const BusinessDetail of business_detail_data) {
      rows.push(BusinessDetail);
    }

    return rows;
  };

  console.log("BusinessDetail dict: ", BusinessDetails_dict()[0]);

  return (
    <div className="flex items-center justify-center w-full overflow-x-auto">
      <Box
        sx={{
          width: "100%",
          height: "calc(100 - 100px)",
          padding: 2,
          "& .MuiDataGrid-root": {
            minWidth: 800, // Minimum width before scrolling
          },
        }}
      >
        <DataGrid
          rows={business_detail_data}
          columns={columns}
          getRowId={(row) => row.id}
          initialState={{
            pagination: {
              paginationModel: {
                pageSize: 10,
              },
            },
          }}
          pageSizeOptions={[10]}
          disableRowSelectionOnClick
          sx={{
            fontSize: "16px",
            boxShadow: 2,
            border: 2,
            borderColor: '#5D76D3',
            backgroundColor: "#FFFFFFFF",
            "& .MuiDataGrid-cell": {
              backgroundColor: "#F4F4F4FF",
              whiteSpace: "normal",
              lineHeight: "1.2",
              padding: "8px",
            },
            "& .MuiDataGrid-columnHeaderTitle": {
              fontWeight: "bold",
            },
            '& .MuiDataGrid-cell:hover': {
              color: '#5D76D3',
            },
          }}
        />
        <BusinessDetailEdit
          open={editModalOpen}
          onClose={handleClose}
          businessDetail={selectedBusinessDetail}
          onSave={handleSave}
          refreshTable={handleRefreshTable}
        />
      </Box>
    </div>
  );
}
