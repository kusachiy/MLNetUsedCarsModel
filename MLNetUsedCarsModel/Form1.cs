using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MLNetUsedCarsModel
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            Worker.Work(Writeline);
        }
        
        private void Writeline(string text)
        {
            this.console.Text += text + '\r'+'\n';
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Worker.Train();
            button2.Enabled = true;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Worker.Evaluate();
            button3.Enabled = true;
            button4.Enabled = true;
            button5.Enabled = true;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            Worker.Test1();
        }

        private void button4_Click(object sender, EventArgs e)
        {
            Worker.Test2();
        }       
        private void button5_Click_1(object sender, EventArgs e)
        {
            Worker.Test3();
        }
    }
}
